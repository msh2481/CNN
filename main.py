# !pip install neptune-client[optuna] qhoptim
import torch
from routines import run, connect_neptune
import optuna
import neptune.new as neptune
import logging
import sys
import neptune.new.integrations.optuna as optuna_utils
import static as st

connect_neptune('mlxa/CNN', 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5NTIzY2UxZC1jMjI5LTRlYTQtYjQ0Yi1kM2JhMGU1NDllYTIifQ==',)
# connect_neptune('mlxa/CNN', None)

neptune_callback = optuna_utils.NeptuneCallback(st.run)

def objective(trial):
    params = {
        'neptune_logging': True,

        'jitter_brightness': 0.01,
        'jitter_contrast': 0.03,
        'jitter_saturation': 0.1,
        'jitter_hue': 0.01,
        'perspective_distortion': 0.1,
        'cutout_count': 0,
        'cutout_min_size': 0,
        'cutout_max_size': 0,

        'model': 'Dummy()',
        'batch_size': 512,
        'k_epoch': 1, 
        'plot_interval': 10,
        'train': 'train_v2.bin',
        'use_per': True,
        'per_alpha': 0.5,
        'per_beta': 0.5,
        'val': 'val_v2.bin',
        'test': None,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',

        'optimizer': 'SGD',
        'lr': 1e-5,
        'beta1': 0.999,
        'wd': 0.0,
        'nesterov': True,
        'lr_scheduler': 'ExponentialLR',
        'gamma': 0.999,
        'epochs': 500,

        'tag': 'underfit2'
    }
    try:
        return run(trial, params)
    except optuna.TrialPruned:
        raise optuna.TrialPruned()
    except Exception as e:
        print('Exception', type(e), e)
        p = neptune.init_project(name='mlxa/CNN', api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5NTIzY2UxZC1jMjI5LTRlYTQtYjQ0Yi1kM2JhMGU1NDllYTIifQ==')
        p['errors'].log({'params': params, 'error': str(e)})
        return 10

optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
study = optuna.create_study(direction='minimize', pruner=optuna.pruners.HyperbandPruner())
study.optimize(objective, n_trials=100, callbacks=[neptune_callback])