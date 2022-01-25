# !pip install neptune-client qhoptim albumentations
import torch
from routines import run

config = {
    'project_name': 'mlxa/CNN',
    'api_token': 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5NTIzY2UxZC1jMjI5LTRlYTQtYjQ0Yi1kM2JhMGU1NDllYTIifQ==',
    'register_run': False,
    'connect_to_project': False,

    'model': 'Dummy()',
    'batch_size': 100,
    'plot_interval': 100,
    'train': 'train_v2.bin',
    'use_per': True,
    'per_alpha': 0.6,
    'per_beta': 0.6,
    'val': 'val_v2.bin',
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
run(config)