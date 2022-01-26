import static as st

import torch.nn as nn
from stats import complex_hash
from tqdm import trange, tqdm
from tianshou.data import Batch, PrioritizedReplayBuffer

def train_epoch(model, dataloader, optimizer, logging=None, interval=None):
    if dataloader is None:
        return
    model.train()
    loss_fn = nn.NLLLoss()
    for batch, (x, y) in (enumerate(dataloader)):
        optimizer.zero_grad()
        # with torch.autocast(dtype=torch.bfloat16, device_type="cpu"):
        x, y = x.to(st.device), y.to(st.device, dtype=int)
        loss = loss_fn(model(st.aug(x)), y)
        loss.backward()
        optimizer.step()
        if logging and (not interval or batch % interval == 0):
            logging(batch, loss.item(), *complex_hash(model, 2))

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

from neptune.new.types import File
import plotly.express as px
from git_utils import save_to_zoo
def train_model_common(model, optimizer, scheduler, epochs, plot_interval):
    global run, train_loader, val_loader, test_loader
    pathx, pathy = [], []
    print(f'started train #{st.run_id}', flush=True)
    for epoch in trange(epochs):
        def train_logging(batch, loss, hx, hy):
            pathx.append(hx)
            pathy.append(hy)
            step = epoch + (batch + 1) / len(st.train_loader)
            st.run['train/epoch'].log(step, step=step)
            st.run['train/train_loss'].log(loss, step=step)
            st.run['train/path'] = File.as_html(px.line(x=pathx, y=pathy))
        def test_logging(loss, acc):
            step = epoch + 1
            st.run['train/epoch'].log(step, step=step)
            st.run['train/val_loss'].log(loss, step=step)
            st.run['train/val_acc'].log(acc, step=step)
            print(f'step: {step}, loss: {loss}, acc: {acc}, hx: {pathx[-1] if pathx else -1}, hy: {pathy[-1] if pathy else -1}')
            name = f'{st.run_id}_{epoch}'
            save_to_zoo(model, name, loss, acc)
            solve_test(model, st.test_loader, f'solution_{model.loader}_{name}')
        train_epoch(model, st.train_loader, optimizer, train_logging, plot_interval)
        scheduler.step()
        with torch.no_grad():
            test_logging(*test(model, st.val_loader))

def train_batch_with_per(model, replay_buffer, optimizer, batch_size):
    if replay_buffer is None:
        return
    model.train()
    loss_fn = nn.NLLLoss(reduction='none')
    optimizer.zero_grad()
    ids = replay_buffer.sample_indices(batch_size)
    d = replay_buffer[ids]
    x = d.obs.to(st.device)
    y = torch.tensor(d.act, device=st.device)
    loss = loss_fn(model(st.aug(x)), y)
    replay_buffer.update_weight(ids, loss)
    loss.mean().backward()
    optimizer.step()
    return loss.mean().item()

def train_model_per(model, optimizer, scheduler, epochs, batches_per_epoch, batch_size, plot_interval):
    global run, train_loader, val_loader, test_loader
    pathx, pathy = [], []
    print(f'started train #{st.run_id}', flush=True)
    for epoch in trange(epochs):           
        for batch in range(batches_per_epoch):
            loss = train_batch_with_per(model, st.train_loader, optimizer, batch_size)
            if batch % plot_interval == 0:
                hx, hy = complex_hash(model, 2)
                pathx.append(hx)
                pathy.append(hy)
                step = epoch + (batch + 1) / batches_per_epoch
                st.run['train/epoch'].log(step, step=step)
                st.run['train/train_loss'].log(loss, step=step)
                st.run['train/path'] = File.as_html(px.line(x=pathx, y=pathy))
        scheduler.step()
        with torch.no_grad():
            loss, acc = test(model, st.val_loader)
            step = epoch + 1
            st.run['train/epoch'].log(step, step=step)
            st.run['train/val_loss'].log(loss, step=step)
            st.run['train/val_acc'].log(acc, step=step)
            print(f'step: {step}, loss: {loss}, acc: {acc}, hx: {pathx[-1] if pathx else -1}, hy: {pathy[-1] if pathy else -1}')
            name = f'{st.run_id}_{epoch}'
            save_to_zoo(model, name, loss, acc)
            solve_test(model, st.test_loader, f'solution_{model.loader}_{name}')

import neptune.new as neptune
from data import Plug, build_dataset, build_model, build_optimizer, build_lr_scheduler
from time import time
from torch.utils.data import DataLoader
from plotly import express as px
from autoaug import build_transforms

def run(config):
    [print(f'{key}: {value}', flush=True) for key, value in config.items()]
    st.device = config['device']
    if config['connect_to_project']:
        st.project = neptune.init_project(name='mlxa/CNN', api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5NTIzY2UxZC1jMjI5LTRlYTQtYjQ0Yi1kM2JhMGU1NDllYTIifQ==')
    else:
        st.project = Plug()

    st.run = neptune.init(project=config['project_name'], api_token=config['api_token']) if config['register_run'] else Plug()
    st.run_id = hex(int(time()))[2:]
    st.run['parameters'] = config

    model = build_model(config)
    optimizer = build_optimizer(model.parameters(), config)
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
    if train_set:
        if config['use_per']:
            train_model_per(model, optimizer, scheduler, config['epochs'], len(train_set) // config['batch_size'], config['batch_size'], config['plot_interval'])
        else:
            train_model_common(model, optimizer, scheduler, config['epochs'], config['plot_interval'])
    save_to_zoo(model, f'{st.run_id}_final', *test(model, st.val_loader))

def norm_rnd(loc, scale, l, r):
    generator = torch.distributions.Normal(loc, scale)
    a = generator.sample(sample_shape=()).item()
    assert l < r
    return max(l, min(r, a))

def uni_rnd(l, r):
    return l + (r - l) * torch.rand(1).item()
from random import randint, choice

def gen_config(epochs):
    cutout_min = norm_rnd(2, 4, 0, 16)
    bs = 64
    return {
        'project_name': 'mlxa/CNN',
        'api_token': 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5NTIzY2UxZC1jMjI5LTRlYTQtYjQ0Yi1kM2JhMGU1NDllYTIifQ==',
        'register_run': True,
        'connect_to_project': True,

        'jitter_brightness': 0.01,
        'jitter_contrast': 0.01,
        'jitter_saturation': 0.01,
        'jitter_hue': 0.01,
        'perspective_distortion': 0.01,
        'cutout_count': int(norm_rnd(0, 1, 0, 10)),
        'cutout_min_size': int(cutout_min),
        'cutout_max_size': int(cutout_min * norm_rnd(2, 0.5, 1, 10)),

        'model': 'M5()',
        'batch_size': bs,
        'plot_interval': (4000 + bs - 1) // bs,
        'train': 'train_v3.bin',
        'use_per': False,
        'val': 'val_v3.bin',
        'test': None,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',

        'optimizer': 'QHAdam',
        'lr': 10**norm_rnd(-4, 1, -6, -1),
        'wd': 10**norm_rnd(-5, 1, -7, -2),
        'beta1': 0.9,
        'beta2': 0.999,
        'nu1': norm_rnd(0.5, 0.2, 0.1, 0.9),
        'nu2': 1,
        'epochs': epochs,

        'tag': 'sweep2'
    }