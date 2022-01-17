import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch import optim
from train import ChessDataset, Net


test_dataset = ChessDataset("TestSet.npz")
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False)

# load model
vals = torch.load("value.pth", map_location=lambda storage, loc: storage)
model = Net()
model.load_state_dict(vals)
model.eval()

correct = 0
total = 0
for boardState, label in test_loader:
    print(boardState.shape)
    outputs = model(boardState)
    score = outputs.data[0][0]
    correct += (score == label).sum()
    total += label.size(0)                    # Increment the total count

print('Accuracy of the network on the 10K test images: %d %%' % (100 * correct / total))

