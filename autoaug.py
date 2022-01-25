from torchvision.transforms import Compose, ColorJitter, RandomPerspective
import torch
from copy import deepcopy
from torch import nn

class Cutout(nn.Module):
	def __init__(self, config):
		super(Cutout, self).__int__()
		self.count = nn.Parameter(config['cutout_count'])
		self.min_size = nn.Parameter(config['cutout_min_size'])
		self.max_size = nn.Parameter(config['cutout_max_size'])
	def forward(tensor):
		assert tensor.dtype == torch.float
		cutted = tensor.detach()
		assert tensor.shape[1:] == (3, 32, 32) and len(tensor.shape) == 4
		mean = tensor.mean(dim=(2, 3)).view(-1, 3, 1, 1)
		for it in range(self.count):
			sz = torch.randint(low=self.min_size, high=self.max_size, size=tuple())
			si = torch.randint(high=tensor.shape[2]-sz, size=tuple())
			sj = torch.randint(high=tensor.shape[3]-sz, size=tuple())
			cutted[:, :, si:si+sz, sj:sj+sz] = mean
		return cutted

def build_transforms(config):
	transforms = [
		ColorJitter(config['jitter_brightness'], config['jitter_contrast'], config['jitter_saturation'], config['jitter_hue']),
		RandomPerspective(distortion_scale=config['perspective_distortion'])
	]
	if config['cutout_count']:
		transforms.append(Cutout(config))
	return nn.Sequential(*transforms)
