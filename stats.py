import torch
import numpy as np

def get_params(model):
    lst = []
    treshold = 100
    for e in model.parameters():
        flat = e.view(-1)
        lst.append(flat[:min(len(flat), treshold)].cpu().detach())
    return torch.cat(lst, dim=0)
def eval_unity_root(poly, arg):
    num = np.exp(1j * arg)
    return np.polyval(poly, num)
def complex_hash(model, n):
    params = get_params(model)
    return np.abs(eval_unity_root(params, np.linspace(0, 2 * np.pi, num = n, endpoint = False)))