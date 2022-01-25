import torch
from os import listdir, system
import static as st


class Plug:
    d = dict()
    def log(self, x, step=None):
        pass
    def upload(self, x):
        pass
    def __getitem__(self, s):
        return self.d[s] if s in self.d else Plug()
    def __setitem__(self, s, v):
        self.d[s] = v

def build_dataset(name):
    import pickle
    if not name:
        return None
    if not name in listdir():
        st.project[name].download(f'{name}')
        assert name in listdir()
    try:
        data = torch.load(name)
    except Exception as e:
        print("torch.load() didn't work, trying pickle")
        with open(name, 'rb') as f:
            data = pickle.load(f)
    return data

from models import *
from git_utils import load_from_zoo
def build_model(config):
    model = eval(config['model'])
    return model.to(config['device'])

from qhoptim.pyt import QHAdam
def build_optimizer(params, config):
    tp = config['optimizer']
    if tp == 'SGD':
        return torch.optim.SGD(params, lr=config['lr'], momentum=config['beta1'], weight_decay=config['wd'])
    elif tp == 'Adam':
        return torch.optim.Adam(params, lr=config['lr'], betas=(config['beta1'], config['beta2']), weight_decay=config['wd'])
    elif tp == 'QHAdam':
        return QHAdam(params, lr=config['lr'], betas=(config['beta1'], config['beta2']), nus=(config['nu1'], config['nu2']), weight_decay=config['wd'])
    else:
        assert False, 'Unknown optimizer'

def build_lr_scheduler(optimizer, config):
    tp = config.get('lr_scheduler')
    if tp is None:
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1)
    elif tp == 'ExponentialLR':
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config['gamma'])
    elif tp == 'OneCycleLR':
        return torch.optim.lr_scheduler.OneCycleLR(optimizer, config['max_lr'], total_steps=config['epochs']*len(st.train_loader), cycle_momentum=False)
    else:
        assert False, 'Unknown LR scheduler'
