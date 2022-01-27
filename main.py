# !pip install neptune-client qhoptim
import torch
from routines import run, gen_config
import optuna
import neptune.new as neptune


def objective(trial):
    cutout_min = trial.suggest_int('cutout_min_size', 2, 8, log=True)
    bs = 64
    params = {
        'project_name': 'mlxa/CNN',
        'api_token': 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5NTIzY2UxZC1jMjI5LTRlYTQtYjQ0Yi1kM2JhMGU1NDllYTIifQ==',
        'register_run': True,
        'connect_to_project': True,

        'jitter_brightness': 0.01,
        'jitter_contrast': 0.01,
        'jitter_saturation': 0.01,
        'jitter_hue': 0.01,
        'perspective_distortion': 0.01,
        'cutout_count': 1,
        'cutout_min_size': cutout_min,
        'cutout_max_size': trial.suggest_int('cutout_max_size', cutout_min, 3 * cutout_min, log=True),

        'model': 'M5()',
        'batch_size': bs,
        'plot_interval': (4000 + bs - 1) // bs,
        'train': 'train_v3.bin',
        'use_per': False,
        'val': 'val_v3.bin',
        'test': None,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',

        'optimizer': 'QHAdam',
        'lr': trial.suggest_float('lr', 1e-6, 1e-2, log=True),
        'wd': trial.suggest_float('wd', 1e-7, 1e-3, log=True)
        'beta1': 0.9,
        'beta2': 0.999,
        'nu1': trial.suggest_float('nu1', 0.1, 0.9),
        'nu2': 1,
        'epochs': 5,

        'tag': 'sweep2'
    }
    try:
        return run(params)
    except:
        p = neptune.init_project(name='mlxa/CNN', api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5NTIzY2UxZC1jMjI5LTRlYTQtYjQ0Yi1kM2JhMGU1NDllYTIifQ==')
        p['errors'].log(params)
        return 10

optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
study = optuna.create_study(pruner=optuna.pruners.Hyperband())
study.optimize(objective, n_trials=20)