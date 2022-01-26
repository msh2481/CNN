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

p = neptune.init_project(name='mlxa/CNN', api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5NTIzY2UxZC1jMjI5LTRlYTQtYjQ0Yi1kM2JhMGU1NDllYTIifQ==')

def smooth(pic):
    assert pic.shape == (3, 32, 32)
    s = torch.zeros_like(pic)
    b = (0 < pic) & (pic < 1)
    g = torch.where(b, pic, torch.zeros(1))
    c = torch.ones_like(pic) * 1e-9
    s[:, 1:, :] += g[:, :-1, :]
    s[:, :-1, :] += g[:, 1:, :]
    s[:, :, 1:] += g[:, :, :-1]
    s[:, :, :-1] += g[:, :, 1:]

    c[:, 1:, :] += b[:, :-1, :]
    c[:, :-1, :] += b[:, 1:, :]
    c[:, :, 1:] += b[:, :, :-1]
    c[:, :, :-1] += b[:, :, 1:]
    return torch.where(b, pic, s/c)

# old_name = 'test_v1.bin'
# new_name = 'test_v3.bin'

# def info(x):
# 	return x.min(), x.max(), x.mean(), x.std()

# data = build_dataset(old_name)
# new = []
# for x, y in tqdm(data):
# 	new.append((smooth(smooth(x)), torch.tensor(y)))
# torch.save(new, new_name)