import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

class IKDataSet(Dataset):
    def __init__(self, file, transform=None):
        # self.ik_frame = pd.read_csv(file)
        self.ik_frame = np.load(file+".npy")
        self.transform = transform
        _min_max = np.load(file+"_min_max.npy")
        self.min = _min_max[0]
        self.max = _min_max[1]

    def __len__(self):
        return len(self.ik_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # xyzrpy = eval(self.ik_frame.loc[idx, 'xyzrpy'])
        # jnt_values = eval(self.ik_frame.loc[idx, 'jnt_values'])
        xyzrpy = self.ik_frame[idx][0]
        xyzrpy = (xyzrpy-self.min)/(self.max-self.min) #normalize
        jnt_values = self.ik_frame[idx][1]
        return torch.Tensor(xyzrpy), torch.Tensor(jnt_values)


class Net(nn.Module):
    def __init__(self, n_hidden, n_jnts):
        super().__init__()
        # self.fc1 = nn.Linear(6, n_hidden)
        # self.fc1_dropout = nn.Dropout(p=.05)
        # self.fc2 = nn.Linear(n_hidden, n_hidden//2)
        # self.fc2_dropout = nn.Dropout(p=.05)
        # self.fc3 = nn.Linear(n_hidden//2, n_hidden//4)
        # self.fc3_dropout = nn.Dropout(p=.05)
        # self.fc4 = nn.Linear(n_hidden//4, n_hidden//8)
        # self.fc4_dropout = nn.Dropout(p=.05)
        # self.fc5 = nn.Linear(n_hidden//8, n_jnts)
        self.fc1 = nn.Linear(6, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden//2)
        self.fc3 = nn.Linear(n_hidden//2, n_jnts)


    def forward(self, x):
        # x = self.fc1(x)
        # x = F.leaky_relu(x, 0.01)
        # x = self.fc1_dropout(x)
        # x = self.fc2(x)
        # x = F.leaky_relu(x, 0.01)
        # x = self.fc2_dropout(x)
        # x = self.fc3(x)
        # x = F.leaky_relu(x, 0.01)
        # x = self.fc3_dropout(x)
        # x = self.fc4(x)
        # x = F.leaky_relu(x, 0.01)
        # x = self.fc4_dropout(x)
        # out = self.fc5(x)
        x = self.fc1(x)
        x = F.leaky_relu(x, 0.01)
        x = self.fc2(x)
        x = F.leaky_relu(x, 0.01)
        out = self.fc3(x)
        return out


def train_loop(dataloader, model, loss_fn, optimizer, device, writer, global_step):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X.to(device=device)
        y.to(device=device)
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        writer.add_scalar('loss', loss.item(), global_step[0])
        writer.add_scalar('grad fc1 weight', torch.norm(model.fc1.weight.grad), global_step[0])
        writer.add_scalar('grad fc2 weight', torch.norm(model.fc2.weight.grad), global_step[0])
        writer.add_scalar('grad fc3 weight', torch.norm(model.fc3.weight.grad), global_step[0])
        global_step[0] += 1
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X.to(device=device)
            y.to(device=device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (torch.abs(pred - y).max(1)[0] < 1e-6).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    device = torch.device('cpu')
    # device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')

    model = Net(n_hidden=1024, n_jnts=6).to(device=device)
    # learning_rate = 1e-3
    learning_rate = 3e-4
    batch_size = 64
    epochs = 20000
    loss_fn = nn.MSELoss()
    # loss_fn = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_data = IKDataSet('data_gen/cobotta_ik')
    test_data = IKDataSet('data_gen/cobotta_ik_test')
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    writer = SummaryWriter()
    global_step = [0]
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer, device, writer, global_step)
    model_path = 'tester/cobotta_model.pth'
    torch.save(model.state_dict(), model_path)
    print("Done!")
    writer.close()
    test_loop(test_dataloader, model, loss_fn, device)
