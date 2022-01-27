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
    cutout_min = trial.suggest_int('cutout_min_size', 2, 8, log=True)
    bs = 64
    params = {
        'neptune_logging': False,

        'jitter_brightness': 0.01,
        'jitter_contrast': 0.01,
        'jitter_saturation': 0.01,
        'jitter_hue': 0.01,
        'perspective_distortion': 0.01,
        'cutout_count': 1,
        'cutout_min_size': cutout_min,
        'cutout_max_size': trial.suggest_int('cutout_max_size', cutout_min + 1, 3 * cutout_min, log=True),

        'model': 'Dummy()',
        'batch_size': bs,
        'plot_interval': (4000 + bs - 1) // bs,
        'train': 'train_v2.bin',
        'use_per': False,
        'val': 'val_v2.bin',
        'test': None,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',

        'optimizer': 'QHAdam',
        'lr': trial.suggest_float('lr', 1e-6, 1e-2, log=True),
        'wd': trial.suggest_float('wd', 1e-7, 1e-3, log=True),
        'beta1': 0.9,
        'beta2': 0.999,
        'nu1': trial.suggest_float('nu1', 0.1, 0.9),
        'nu2': 1,
        'epochs': 5,

        'tag': 'sweep3'
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
study.optimize(objective, n_trials=10, callbacks=[neptune_callback])