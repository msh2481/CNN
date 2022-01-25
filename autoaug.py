from torchvision.transforms import autoaugment, AutoAugment, Compose
import albumentations as A
import torch
import cv2
from copy import deepcopy

# picture_autoaug = AutoAugment(autoaugment.AutoAugmentPolicy.SVHN)
def autoaug(tensor):
	assert tensor.dtype == torch.float
	# return picture_autoaug(image=(255 * tensor).to(torch.uint8)).to(torch.float) / 255
	# return torch.tensor(picture_autoaug(image=tensor.numpy()))
	cutted = deepcopy(tensor)
	assert tensor.shape[1:] == (3, 32, 32) and len(tensor.shape) == 4
	mean = tensor.mean(dim=(2, 3)).view(-1, 3, 1, 1)
	for it in range(1):
		sz = torch.randint(low=8, high=16, size=tuple())
		si = torch.randint(high=tensor.shape[2]-sz, size=tuple())
		sj = torch.randint(high=tensor.shape[3]-sz, size=tuple())
		cutted[:, :, si:si+sz, sj:sj+sz] = mean
	return cutted