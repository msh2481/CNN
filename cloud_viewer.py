import neptune.new as neptune
import torch
from data import build_dataset
from tqdm import trange, tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import ToTensor, Compose
from torchvision.datasets.vision import VisionDataset
import os
import pickle
from typing import Any, Callable, Optional, Tuple
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from plotly import express as px

p = neptune.init_project(name='mlxa/CNN', api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5NTIzY2UxZC1jMjI5LTRlYTQtYjQ0Yi1kM2JhMGU1NDllYTIifQ==')

# def smooth(pic):
#     assert pic.shape == (3, 32, 32)
#     s = torch.zeros_like(pic)
#     b = (0 < pic) & (pic < 1)
#     g = torch.where(b, pic, torch.zeros(1))
#     c = torch.ones_like(pic) * 1e-9
#     s[:, 1:, :] += g[:, :-1, :]
#     s[:, :-1, :] += g[:, 1:, :]
#     s[:, :, 1:] += g[:, :, :-1]
#     s[:, :, :-1] += g[:, :, 1:]

#     c[:, 1:, :] += b[:, :-1, :]
#     c[:, :-1, :] += b[:, 1:, :]
#     c[:, :, 1:] += b[:, :, :-1]
#     c[:, :, :-1] += b[:, :, 1:]
#     return torch.where(b, pic, s/c)

# old_name = 'test_v1.bin'
# new_name = 'test_v3.bin'

# def info(x):
# 	return x.min(), x.max(), x.mean(), x.std()

# data = build_dataset(old_name)
# new = []
# for x, y in tqdm(data):
# 	new.append((smooth(smooth(x)), torch.tensor(y)))
# torch.save(new, new_name)

# runs_table_df = p.fetch_runs_table().to_pandas()
# a = runs_table_df[['sys/id', 'parameters/batch_size', 'parameters/beta1', 'parameters/beta2',
#        'parameters/connect_to_project', 'parameters/cutout_count',
#        'parameters/cutout_max_size', 'parameters/cutout_min_size',
#        'parameters/device', 'parameters/dropout', 'parameters/epochs',
#        'parameters/gamma', 'parameters/jitter_brightness',
#        'parameters/jitter_contrast', 'parameters/jitter_hue',
#        'parameters/jitter_saturation', 'parameters/lr',
#        'parameters/lr_scheduler', 'parameters/max_lr', 'parameters/model',
#        'parameters/nu1', 'parameters/nu2', 'parameters/optimizer',
#        'parameters/per_alpha', 'parameters/per_beta',
#        'parameters/perspective_distortion', 'parameters/plot_interval',
#        'parameters/project_name', 'parameters/register_run', 'parameters/test',
#        'parameters/train', 'parameters/use_per', 'parameters/val',
#        'parameters/wd', 'train/val_acc', 'train/val_loss']]
# print(a)
# a.to_csv('sweeps/sweep1.csv')

import pandas as pd
a = pd.read_csv('sweeps/sweep1.csv')
treshold = 0.8
good = a[a['train/val_acc']>=treshold]
bad =  a[a['train/val_acc']<treshold]
print(good)
print(bad)
for param in a.columns:
	if 'parameters/' not in param:
		continue
	try:
		print(f'{param}: {good[param].median()} vs {bad[param].median()}')
	except:
		continue