import torch
from train import Net

vals = torch.load("value.pth", map_location=lambda storage, loc: storage)
model = Net()
model.load_state_dict(vals)




