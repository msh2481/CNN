import static as st
from os import system, listdir
import torch
from models import *

def save_to_zoo(model, name, val_loss=None, val_acc=None):
    print(f'save_to_zoo {model.loader}_{name}.p, acc: {val_acc}, loss: {val_loss}')
    model.cpu()
    torch.save({
        'state_dict': model.state_dict(),
        'loader': model.loader,
        'val_loss': val_loss,
        'val_acc': val_acc,
        }, f'model.p')
    model.to(st.device)
    st.project[f'zoo/{model.loader}_{name}.p'].upload(f'model.p')

def load_from_zoo(name):
    st.project[f'zoo/{name}'].download(name)
    d = torch.load(f'{name}', map_location='cpu')
    model = eval(d['loader'])
    model.load_state_dict(d['state_dict'])
    return model.to(st.device)
