import static as st

import torch
import torch.nn as nn
from stats import complex_hash
from tqdm import trange, tqdm
from torch.nn import functional as F

def get_output(model, data):
    results = []
    step = 100
    for i in range(0, len(data), step):
        batch = data[i : min(i + 100, len(data))]
        results.append(model(batch.to(st.device)))
    return torch.cat(results, dim=0).cpu()

def test(model, dataset):
    if dataset is None:
        return
    model.eval()
    x, y = dataset
    p = get_output(model, x)
    loss = F.nll_loss(p, y)
    acc = (p.argmax(dim=-1) == y).float().mean()
    return loss.item(), acc.item()

def write_solution(filename, labels):
    with open(filename, 'w') as solution:
        print('Id,Category', file=solution)
        for i, label in enumerate(labels):
            print(f'{i},{label}', file=solution)

def solve_test(model, dataset, name):
    if dataset is None:
        return
    model.eval()
    with torch.no_grad():
        predictions = get_output(model, dataset).numpy()
    print('saved', len(predictions), 'predictions to ', name)
    write_solution(f'{name}.csv', predictions)

import torch.distributions as dist
def train_epoch(model, dataset, optimizer, n_batches, batch_size, alpha, beta, flood_level, logging, plot_interval):
    data, targets = dataset
    with torch.no_grad():
        outputs = get_output(model, data)
        loss = F.nll_loss(outputs, targets, reduction='none')
        assert loss.shape == (len(data), )
        probs = F.softmax(loss * alpha, dim=-1)
        batch_indices = torch.multinomial(probs, n_batches*batch_size, replacement=True).view(n_batches, batch_size)
        importance_sampling_weights = 1/(probs[batch_indices]*len(data))
        importance_sampling_weights /= importance_sampling_weights.sum(dim=1, keepdim=True)
        importance_sampling_weights = importance_sampling_weights.to(st.device)
    acc = 0
    for batch_idx in range(n_batches):
        x = data[batch_indices[batch_idx]].to(st.device)
        y = targets[batch_indices[batch_idx]].to(st.device)
        optimizer.zero_grad()
        outputs = model(st.aug(x))
        raw_loss = F.nll_loss(outputs, y, reduction='none')
        loss = (raw_loss * importance_sampling_weights[batch_idx]).mean()
        loss = (loss - flood_level).abs() + flood_level
        acc += (outputs.argmax(dim=-1) == y).float().mean().item()
        loss.backward()
        optimizer.step()
        if batch_idx % plot_interval == plot_interval - 1:
            logging((batch_idx + 1) / n_batches, loss.mean().item(), acc / plot_interval, *complex_hash(model, 2))
            acc = 0

from neptune.new.types import File
import plotly.express as px
from git_utils import save_to_zoo
import optuna

def train_model(trial, model, optimizer, scheduler, config):
    pathx, pathy = [], []
    min_loss = 1e9
    st.run_id = hex(int(time()))[2:]
    print(f'started train #{st.run_id}', flush=True)
    val_loss, val_acc = 0.0, 0.0
    for epoch in trange(config['epochs']):
        def train_logging(batch_pos, loss, acc, hx, hy):
            pathx.append(hx)
            pathy.append(hy)
            step = epoch + batch_pos
            if config['neptune_logging']:
                st.run['train/epoch'].log(step, step=step)
                st.run['train/train_loss'].log(loss, step=step)
                st.run['train/train_acc'].log(acc, step=step)
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
            if trial:
                trial.report(min_loss, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            name = f'{st.run_id}_{epoch}'
            save_to_zoo(model, name, loss, acc)
        train_epoch(
            model,
            st.train_set,
            optimizer,
            config['k_epoch'] * len(st.train_set[0]) // config['batch_size'] + 1,
            config['batch_size'],
            config['per_alpha'],
            config['per_beta'],
            torch.normal(mean=config['flood_mean'], std=config['flood_std'], size=(1,)).item(),
            train_logging,
            config['plot_interval'])
        scheduler.step()
        with torch.no_grad():
            val_loss, val_acc = test(model, st.val_set)
            test_logging(val_loss, val_acc)
    return min_loss

from data import Plug, build_dataset, build_model, build_optimizer, build_lr_scheduler
from time import time
from plotly import express as px
from autoaug import build_transforms
import neptune.new as neptune

def connect_neptune(project_name, run_token):
    st.project = neptune.init_project(name='mlxa/CNN', api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5NTIzY2UxZC1jMjI5LTRlYTQtYjQ0Yi1kM2JhMGU1NDllYTIifQ==')
    st.run = neptune.init(project=project_name, api_token=run_token) if run_token else Plug()

def run(trial, config):
    [print(f'{key}: {value}', flush=True) for key, value in config.items()]
    st.device = config['device']
    st.run['parameters'] = config

    model = build_model(config)
    optimizer = build_optimizer(model.parameters(), config)
    st.train_set = build_dataset(config['train'])
    st.aug = build_transforms(config)
    # pics = st.aug(torch.stack([e[0] for e in train_set[:5]]))
    # for pic in pics:
    #     pic = pic.clip(0, 1)
    #     px.imshow(pic.permute(1, 2, 0)).show()
    scheduler = build_lr_scheduler(optimizer, config)
    st.val_set  = build_dataset(config['val'])
    st.test_set = build_dataset(config['test'])
    result = train_model(trial, model, optimizer, scheduler, config) if st.train_set else None
    # save_to_zoo(model, f'{st.run_id}_final', *test(model, st.val_set))
    solve_test(model, st.test_set, f'solution_{model.loader}_{st.run_id}')
    return result