# !pip install neptune-client qhoptim
import torch
from routines import run, gen_config

config = {
    'project_name': 'mlxa/CNN',
    'api_token': 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5NTIzY2UxZC1jMjI5LTRlYTQtYjQ0Yi1kM2JhMGU1NDllYTIifQ==',
    'register_run': False,
    'connect_to_project': False,

    'jitter_brightness': 0.1,
    'jitter_contrast': 0.1,
    'jitter_saturation': 0.1,
    'jitter_hue': 0.1,
    'perspective_distortion': 0.1,
    'cutout_count': 1,
    'cutout_min_size': 8,
    'cutout_max_size': 16,

    'model': 'Dummy()',
    'batch_size': 100,
    'plot_interval': 100,
    'train': 'train_v3.bin',
    'use_per': False,
    'val': 'val_v3.bin',
    'test': None,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',

    'optimizer': 'QHAdam',
    'lr': 0.001,
    'wd': 1e-6,
    'dropout': 0.1,
    'beta1': 0.9,
    'beta2': 0.999,
    'nu1': 0.7,
    'nu2': 1.0,
    'epochs': 50
}

run(gen_config(2))