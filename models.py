import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms

class Resnet18(nn.Module):
    def __init__(self, fc_size):
        from torchvision.models import resnet18
        super(Resnet18, self).__init__()
        self.loader = f'Resnet18({fc_size})'
        self.net = resnet18()
        self.net.fc = nn.Linear(512, fc_size)
    def get_logits(self, x):
        return self.net(x)
    def forward(self, x):
        return F.log_softmax(self.get_logits(x), dim=-1)

class Densenet121(nn.Module):
    def __init__(self, fc_size):
        from torchvision.models import densenet121
        super(Densenet121, self).__init__()
        self.loader = f'Densenet121({fc_size})'
        self.net = densenet121()
        self.net.classifier = nn.Linear(1024, fc_size)
    def get_logits(self, x):
        return self.net(x)
    def forward(self, x):
        return F.log_softmax(self.get_logits(x), dim=-1)

class Densenet161(nn.Module):
    def __init__(self, fc_size):
        from torchvision.models import densenet161
        super(Densenet161, self).__init__()
        self.loader = f'Densenet161({fc_size})'
        self.net = densenet161()
        self.net.classifier = nn.Linear(2208, fc_size)
    def get_logits(self, x):
        return self.net(x)
    def forward(self, x):
        return F.log_softmax(self.get_logits(x), dim=-1)

class WideResnet50(nn.Module):
    def __init__(self, fc_size):
        from torchvision.models import wide_resnet50_2
        super(WideResnet50, self).__init__()
        self.loader = f'WideResnet50({fc_size})'
        self.net = wide_resnet50_2()
        self.net.fc = nn.Linear(2048, fc_size)
    def get_logits(self, x):
        return self.net(x)
    def forward(self, x):
        return F.log_softmax(self.get_logits(x), dim=-1)

import torch
import torch.nn as nn
import torch.nn.functional as F

class M3(nn.Module):
    def __init__(self):
        super(M3, self).__init__()
        self.loader = 'M3()'
        self.crop = transforms.CenterCrop(28)
        self.conv1 = nn.Conv2d(3, 32, 3, bias=False)       # output becomes 26x26
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 48, 3, bias=False)      # output becomes 24x24
        self.conv2_bn = nn.BatchNorm2d(48)
        self.conv3 = nn.Conv2d(48, 64, 3, bias=False)      # output becomes 22x22
        self.conv3_bn = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 80, 3, bias=False)      # output becomes 20x20
        self.conv4_bn = nn.BatchNorm2d(80)
        self.conv5 = nn.Conv2d(80, 96, 3, bias=False)      # output becomes 18x18
        self.conv5_bn = nn.BatchNorm2d(96)
        self.conv6 = nn.Conv2d(96, 112, 3, bias=False)     # output becomes 16x16
        self.conv6_bn = nn.BatchNorm2d(112)
        self.conv7 = nn.Conv2d(112, 128, 3, bias=False)    # output becomes 14x14
        self.conv7_bn = nn.BatchNorm2d(128)
        self.conv8 = nn.Conv2d(128, 144, 3, bias=False)    # output becomes 12x12
        self.conv8_bn = nn.BatchNorm2d(144)
        self.conv9 = nn.Conv2d(144, 160, 3, bias=False)    # output becomes 10x10
        self.conv9_bn = nn.BatchNorm2d(160)
        self.conv10 = nn.Conv2d(160, 176, 3, bias=False)   # output becomes 8x8
        self.conv10_bn = nn.BatchNorm2d(176)
        self.fc1 = nn.Linear(25344, 10, bias=False)
        self.fc1_bn = nn.BatchNorm1d(10)
    def get_logits(self, x):
        x = self.crop((x - 0.5) * 2.0)
        conv1 = F.relu(self.conv1_bn(self.conv1(x)))
        conv2 = F.relu(self.conv2_bn(self.conv2(conv1)))
        conv3 = F.relu(self.conv3_bn(self.conv3(conv2)))
        conv4 = F.relu(self.conv4_bn(self.conv4(conv3)))
        conv5 = F.relu(self.conv5_bn(self.conv5(conv4)))
        conv6 = F.relu(self.conv6_bn(self.conv6(conv5)))
        conv7 = F.relu(self.conv7_bn(self.conv7(conv6)))
        conv8 = F.relu(self.conv8_bn(self.conv8(conv7)))
        conv9 = F.relu(self.conv9_bn(self.conv9(conv8)))
        conv10 = F.relu(self.conv10_bn(self.conv10(conv9)))
        flat1 = torch.flatten(conv10.permute(0, 2, 3, 1), 1)
        logits = self.fc1_bn(self.fc1(flat1))
        return logits
    def forward(self, x):
        return F.log_softmax(self.get_logits(x), dim=-1)

class M5(nn.Module):
    def __init__(self):
        super(M5, self).__init__()
        self.loader = 'M5()'
        self.crop = transforms.CenterCrop(28)
        self.conv1 = nn.Conv2d(3, 32, 5, bias=False)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 5, bias=False)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 96, 5, bias=False)
        self.conv3_bn = nn.BatchNorm2d(96)
        self.conv4 = nn.Conv2d(96, 128, 5, bias=False)
        self.conv4_bn = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 160, 5, bias=False)
        self.conv5_bn = nn.BatchNorm2d(160)
        self.fc1 = nn.Linear(10240, 10, bias=False)
        self.fc1_bn = nn.BatchNorm1d(10)
    def get_logits(self, x):
        x = self.crop((x - 0.5) * 2.0)
        conv1 = F.relu(self.conv1_bn(self.conv1(x)))
        conv2 = F.relu(self.conv2_bn(self.conv2(conv1)))
        conv3 = F.relu(self.conv3_bn(self.conv3(conv2)))
        conv4 = F.relu(self.conv4_bn(self.conv4(conv3)))
        conv5 = F.relu(self.conv5_bn(self.conv5(conv4)))
        flat5 = torch.flatten(conv5.permute(0, 2, 3, 1), 1)
        logits = self.fc1_bn(self.fc1(flat5))
        return logits
    def forward(self, x):
        return F.log_softmax(self.get_logits(x), dim=-1)

class M7(nn.Module):
    def __init__(self):
        super(M7, self).__init__()
        self.loader = 'M7()'
        self.crop = transforms.CenterCrop(28)
        self.conv1 = nn.Conv2d(3, 48, 7, bias=False)    # output becomes 22x22
        self.conv1_bn = nn.BatchNorm2d(48)
        self.conv2 = nn.Conv2d(48, 96, 7, bias=False)   # output becomes 16x16
        self.conv2_bn = nn.BatchNorm2d(96)
        self.conv3 = nn.Conv2d(96, 144, 7, bias=False)  # output becomes 10x10
        self.conv3_bn = nn.BatchNorm2d(144)
        self.conv4 = nn.Conv2d(144, 192, 7, bias=False) # output becomes 4x4
        self.conv4_bn = nn.BatchNorm2d(192)
        self.fc1 = nn.Linear(3072, 10, bias=False)
        self.fc1_bn = nn.BatchNorm1d(10)
    def get_logits(self, x):
        x = self.crop((x - 0.5) * 2.0)
        conv1 = F.relu(self.conv1_bn(self.conv1(x)))
        conv2 = F.relu(self.conv2_bn(self.conv2(conv1)))
        conv3 = F.relu(self.conv3_bn(self.conv3(conv2)))
        conv4 = F.relu(self.conv4_bn(self.conv4(conv3)))
        flat1 = torch.flatten(conv4.permute(0, 2, 3, 1), 1)
        logits = self.fc1_bn(self.fc1(flat1))
        return logits
    def forward(self, x):
        return F.log_softmax(self.get_logits(x), dim=-1)

class M7S1(nn.Module):
    def __init__(self):
        super(M7S1, self).__init__()
        self.loader = 'M7S1()'
        self.crop = transforms.CenterCrop(28)
        self.conv1 = nn.Conv2d(3, 8, 7, bias=False)    # output becomes 22x22
        self.conv1_bn = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, 7, bias=False)   # output becomes 16x16
        self.conv2_bn = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 16, 7, bias=False)  # output becomes 10x10
        self.conv3_bn = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(16, 32, 7, bias=False) # output becomes 4x4
        self.conv4_bn = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(512, 10, bias=False)
        self.fc1_bn = nn.BatchNorm1d(10)
    def get_logits(self, x):
        x = self.crop((x - 0.5) * 2.0)
        conv1 = F.relu(self.conv1_bn(self.conv1(x)))
        conv2 = F.relu(self.conv2_bn(self.conv2(conv1)))
        conv3 = F.relu(self.conv3_bn(self.conv3(conv2)))
        conv4 = F.relu(self.conv4_bn(self.conv4(conv3)))
        flat1 = torch.flatten(conv4.permute(0, 2, 3, 1), 1)
        logits = self.fc1_bn(self.fc1(flat1))
        return logits
    def forward(self, x):
        return F.log_softmax(self.get_logits(x), dim=-1)

class Dummy(nn.Module):
    def __init__(self):
        super(Dummy, self).__init__()
        self.loader = 'Dummy()'
        self.line = nn.Linear(1, 10)
    def get_logits(self, x):
        return self.line(x.sum(dim=(-3, -2, -1)).view(-1, 1))
    def forward(self, x):
        return F.log_softmax(self.get_logits(x), dim=-1)

class MixedEnsemble(nn.Module):
    def __init__(self, models, freeze_models):
        super(MixedEnsemble, self).__init__()
        self.loader = 'MixedEnsemble([], [])'
        self.M = len(models)
        if freeze_models:
            for model in models:
                for param in model.parameters():
                    param.requires_grad = False
        self.models = nn.ModuleList(models)
        self.coefs = nn.ParameterList([nn.Parameter(torch.tensor(1.0)) for m in models])
    def forward(self, x):
        return (torch.stack([self.models[i](x).exp() * self.coefs[i] for i in range(self.M)]).sum(dim=0) / sum(self.coefs)).log()

class VotingEnsemble(nn.Module):
    def __init__(self, models, coefs):
        assert len(models) == len(coefs)
        super(VotingEnsemble, self).__init__()
        self.loader = 'MixedEnsemble([], [])'
        self.M = len(models)
        self.models = nn.ModuleList(models)
        self.coefs = nn.ParameterList([nn.Parameter(torch.tensor(c, dtype=torch.float)) for c in coefs])
        for param in self.parameters():
                param.requires_grad = False
    def forward(self, x):
        self.eval()
        print('x:', x.shape, x.sum(), x.std())
        ans = torch.zeros_like(self.models[0](x), dtype=torch.float)
        for i, m in enumerate(self.models):
            pos = m(x).argmax(dim=-1)
            for j in range(x.shape[0]):
                ans[j][pos[j]] += self.coefs[i]
        print('ans:', ans.shape, ans.sum(), ans.std())
        print('ans[0]:', ans[0])
        return ans / self.M