import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch import optim

class ChessDataset(Dataset):
    def __init__(self, inFile) -> None:
        data = np.load(inFile)
        self.X = data['arr_0']
        self.Y = data['arr_1']
        print("Loaded %d samples" % self.X.shape[0])

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return (self.X[index], self.Y[index])

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.stack = nn.Sequential(
                nn.Conv2d(13, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2),
                nn.ReLU(),
                # nn.Dropout2d(),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=3, stride=2),
                nn.ReLU(),
                # nn.Dropout2d(),
                nn.Conv2d(128, 128, kernel_size=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=2, stride=2),
                nn.ReLU(),
                # nn.Dropout2d(),
                nn.Conv2d(256, 256, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(256, 1))

    def forward(self, x):
        x = self.stack(x)
        # value output
        return torch.tanh(x)

if __name__ ==  '__main__':
    chess_dataset = ChessDataset("data_25k.npz")
    train_loader = torch.utils.data.DataLoader(chess_dataset, batch_size=512, shuffle=True)

    model = Net()
    optimizer = optim.Adam(model.parameters())
    floss = nn.MSELoss()

    device = 'cuda'
    model.cuda()

    model.train()

    for epoch in range(100):
        all_loss = 0
        num_loss = 0

        for i, (data, target) in enumerate(train_loader):
            target = target.unsqueeze(-1)
            data, target = data.to(device), target.to(device)
            data = data.float()
            target = target.float()

            optimizer.zero_grad()
            output = model(data)

            loss = floss(output, target)
            loss.backward()
            optimizer.step()

            all_loss += loss.item()
            num_loss += 1

        print('%3d: %f' % (epoch, all_loss/num_loss))

    torch.save(model.state_dict(), 'value.pth')
