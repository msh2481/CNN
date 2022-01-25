import neptune.new as neptune
from inspect import *
import torch

p = neptune.init_project(name='mlxa/CNN', api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5NTIzY2UxZC1jMjI5LTRlYTQtYjQ0Yi1kM2JhMGU1NDllYTIifQ==')
l = list(p.get_structure()['zoo'])
for name in l:
	if 'Wide' in name:
		continue
	if 'Res' not in name:
		continue
	# p[f'zoo/{name}'].download('model.p')
	# try:
	# 	d = torch.load('model.p', map_location='cpu')
	# 	print(name)
	# 	print(d['val_acc'], d['val_loss'])
	# 	if d['val_acc'] < 0.5:
	# 		print('deleting')
	# 		del p[f'zoo/{name}']
	# except Exception:
	# 	print("can't read", name)
	print(name)