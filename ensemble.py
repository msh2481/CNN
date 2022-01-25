from os import chdir, listdir
from torchvision import models
import torch
import torch.nn as nn
from data import build_dataloaders, device

torch.cuda.empty_cache()
# train_loader, val_loader, test_loader = build_dataloaders(batch_size=64, download=False)

def make_predictions(models, x, coefs):
    assert coefs is not None
    x = x.cuda()
    s = 0
    for model, k in zip(models, coefs):
        model.cuda()
        pred = model(x).cpu()
        model.cpu()
        s += pred * k
    return s

def test(models, dataloader, coefs, loss_fn=nn.CrossEntropyLoss()):
    for model in models:
        model.eval()
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    for x, y in dataloader:
        pred = make_predictions(models, x, coefs)
        test_loss += loss_fn(pred, y)
        correct += (pred.argmax(dim=-1) == y).to(torch.float).mean()
    return test_loss / num_batches, correct / num_batches

# chdir('zoo')
zoo = []

for name in listdir():
    if name[-2:] != '.p':
        continue
    if 'resnet' in name:
        model = models.resnet18()
        model.fc = nn.Linear(512, 10)
        model.load_state_dict(torch.load(name, map_location=device))
    else:
        model = torch.load(name, map_location=device)
    model = model.cpu()
    zoo.append(model)
    print(name, flush=True)
    # print(name, test([model], val_loader, None), flush=True)

best = torch.rand(len(zoo))
bval = test(zoo, val_loader, best)[1]

with torch.no_grad():
    for i in range(10**9):
        coefs = torch.rand(len(models))
        loss, acc = test(zoo, val_loader, coefs)
        print('l', loss.item(), 'bl', bval)
        print('c', *coefs.numpy(), flush=True)
        print('b', *best.numpy(), flush=True)
        if bval > acc:
            bval = acc
            best = coefs