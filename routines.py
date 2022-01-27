import static as st

import torch.nn as nn
from stats import complex_hash
from tqdm import trange, tqdm
from tianshou.data import Batch, PrioritizedReplayBuffer

def test(model, dataloader, loss_fn=nn.NLLLoss()):
    if dataloader is None:
        return
    model.eval()
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    for x, y in dataloader:
        x, y = x.to(st.device), y.to(st.device, dtype=int)
        pred = model(x)
        test_loss = test_loss + loss_fn(pred, y)
        correct += (pred.argmax(dim=-1) == y).to(torch.float).mean()
    return (test_loss / num_batches).item(), (correct / num_batches).item()

def write_solution(filename, labels):
    with open(filename, 'w') as solution:
        print('Id,Category', file=solution)
        for i, label in enumerate(labels):
            print(f'{i},{label}', file=solution)

import torch
def solve_test(model, dataloader, name):
    if dataloader is None:
        return
    model.eval()
    predictions = []
    with torch.no_grad():
        for x, _ in dataloader:
            pred = model(x.to(st.device))
            predictions.extend(list(pred.argmax(dim=-1).cpu().numpy()))
    print('saved', len(predictions), 'predictions to ', name)
    write_solution(f'{name}.csv', predictions)

def train_epoch_without_per(model, dataloader, optimizer, logging, plot_interval):
    if dataloader is None:
        return
    debug('epoch in')
    model.train()
    loss_fn = nn.NLLLoss()
    for batch, (x, y) in enumerate(dataloader):
        optimizer.zero_grad()
        x, y = x.to(st.device), y.to(st.device, dtype=int)
        debug('before loss')
        loss = loss_fn(model(st.aug(x)), y)
        debug('after loss')
        loss.backward()
        optimizer.step()
        if batch % plot_interval == 0:
            logging(batch, loss.item(), *complex_hash(model, 2))
    debug('epoch out')
def train_epoch_with_per(model, replay_buffer, optimizer, batches_per_epoch, batch_size, logging, plot_interval):
    if replay_buffer is None:
        return
    model.train()
    loss_fn = nn.NLLLoss(reduction='none')
    for batch in range(batches_per_epoch):
        optimizer.zero_grad()
        ids = replay_buffer.sample_indices(batch_size)
        d = replay_buffer[ids]
        x = d.obs.to(st.device)
        y = torch.tensor(d.act, device=st.device)
        loss_tensor = loss_fn(model(st.aug(x)), y)
        replay_buffer.update_weight(ids, loss_tensor)
        loss = loss_tensor.mean()
        loss.backward()
        optimizer.step()
        if batch % plot_interval == 0:
            logging(batch, loss.item(), *complex_hash(model, 2))

from neptune.new.types import File
import plotly.express as px
from git_utils import save_to_zoo
import optuna

def train_model(trial, model, optimizer, scheduler, config):
    pathx, pathy = [], []
    min_loss = 1e9
    st.run_id = hex(int(time()))[2:]
    print(f'started train #{st.run_id}', flush=True)
    for epoch in trange(config['epochs']):
        def train_logging(batch, loss, hx, hy):
            pathx.append(hx)
            pathy.append(hy)
            step = epoch + (batch + 1) / len(st.train_loader)
            if config['neptune_logging']:
                st.run['train/epoch'].log(step, step=step)
                st.run['train/train_loss'].log(loss, step=step)
                st.run['train/path'] = File.as_html(px.line(x=pathx, y=pathy))
        def test_logging(loss, acc):
            nonlocal min_loss
            step = epoch + 1
            if config['neptune_logging']:
                st.run['train/epoch'].log(step, step=step)
                st.run['train/val_loss'].log(loss, step=step)
                st.run['train/val_acc'].log(acc, step=step)
            print(f'step: {step}, loss: {loss}, acc: {acc}, hx: {pathx[-1] if pathx else -1}, hy: {pathy[-1] if pathy else -1}')
            min_loss = min(min_loss, loss)
            trial.report(min_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
            name = f'{st.run_id}_{epoch}'
            save_to_zoo(model, name, loss, acc)
        if config['use_per']:
            train_epoch_with_per(model, st.train_loader, optimizer, len(train_loader) // config['batch_size'], config['batch_size'], train_logging, config['plot_interval'])
        else:
            train_epoch_without_per(model, st.train_loader, optimizer, train_logging, config['plot_interval'])
        scheduler.step()
        with torch.no_grad():
            test_logging(*test(model, st.val_loader))
    return min_loss

from data import Plug, build_dataset, build_model, build_optimizer, build_lr_scheduler
from time import time
from torch.utils.data import DataLoader
from plotly import express as px
from autoaug import build_transforms
import neptune.new as neptune

def debug(message):
    print(message, flush=True)

def connect_neptune(project_name, run_token):
    st.project = neptune.init_project(name='mlxa/CNN', api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5NTIzY2UxZC1jMjI5LTRlYTQtYjQ0Yi1kM2JhMGU1NDllYTIifQ==')
    st.run = neptune.init(project=project_name, api_token=run_token) if run_token else Plug()

def run(trial, config):
    [print(f'{key}: {value}', flush=True) for key, value in config.items()]
    st.device = config['device']
    st.run['parameters'] = config

    model = build_model(config)
    debug('built model')
    optimizer = build_optimizer(model.parameters(), config)
    debug('built optimizer')
    train_set = build_dataset(config['train'])
    st.aug = build_transforms(config)
    # pics = st.aug(torch.stack([e[0] for e in train_set[:5]]))
    # for pic in pics:
    #     pic = pic.clip(0, 1)
    #     px.imshow(pic.permute(1, 2, 0)).show()
    if config['use_per']:
        if train_set:
            st.train_loader = PrioritizedReplayBuffer(size=len(train_set), alpha=config['per_alpha'], beta=config['per_beta'])
            for x, y in train_set:
                st.train_loader.add(Batch(obs=x, act=y, rew=0, done=False))
        else:
            st.train_loader = None
    else:
        st.train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True) if train_set else None
    scheduler = build_lr_scheduler(optimizer, config)
    val_set, test_set = build_dataset(config['val']), build_dataset(config['test'])
    st.val_loader = DataLoader(val_set, batch_size=config['batch_size'], shuffle=False) if val_set else None
    st.test_loader = DataLoader(test_set, batch_size=config['batch_size'], shuffle=False) if test_set else None
    result = train_model(trial, model, optimizer, scheduler, config) if train_set else None
    # save_to_zoo(model, f'{st.run_id}_final', *test(model, st.val_loader))
    solve_test(model, st.test_loader, f'solution_{model.loader}_{st.run_id}')
    return result