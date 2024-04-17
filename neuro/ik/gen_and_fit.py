import numpy as np
import pandas as pd
import torch.nn as nn
import torch
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from scipy.transform import Rotation


def gen_data(robot, granularity, save_name='ik_data.csv'):
    data_set = []
    start = np.floor(robot.jnt_ranges[:, 0] / granularity)
    end = np.ceil(robot.jnt_ranges[:, 1] / granularity)
    n_data_vec = (end-start+1).astype(int)
    n_data = np.prod(n_data_vec)
    in_data_npy = np.empty((0, robot.n_dof))
    out_data_npy = np.empty((0, 6))
    for i in range(n_data):
        jnt_values = robot.rand_conf()
        tgt_pos, tgt_rotmat = robot.fk(jnt_values=jnt_values, toggle_jacobian=False, update=False)
        tgt_wvec = Rotation.from_matrix(tgt_rotmat).as_rotvec()
        in_data_npy = np.vstack((in_data_npy, jnt_values))
        out_data_npy = np.vstack((out_data_npy, np.hstack((tgt_pos, tgt_wvec))))
        data_set.append([in_data_npy, out_data_npy])
    np.save(save_name, np.array(data_set))


class IKDataSet(Dataset):
    def __init__(self, file):
        self.ik_frame = np.load(file + ".npy")

    def __len__(self):
        return len(self.ik_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        xyzwvec = self.ik_frame[idx][0]
        jnt_values = self.ik_frame[idx][1]
        return torch.Tensor(xyzwvec), torch.Tensor(jnt_values)


class Net(nn.Module):
    def __init__(self, n_hidden, n_jnts):
        super().__init__()
        self.fc1 = nn.Linear(6, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_hidden)
        self.fc4 = nn.Linear(n_hidden, n_jnts)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        out = self.fc4(x)
        return out


def train_loop(dataloader, model, loss_fn, optimizer, writer, global_step):
    for inputs, targets in dataloader:
        # Compute prediction and loss
        pred = model(inputs)
        loss = loss_fn(pred, targets)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        writer.add_scalar('loss', loss.item(), global_step[0])
        global_step[0] += 1
        print(f'    Loss: {loss.item()}')


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            pred = model(inputs)
            test_loss += loss_fn(pred, targets).item()
            correct += (torch.abs(pred - y).max(1)[0] < 1e-6).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    # import os
    #
    # os.system('tensorboard --logdir=runs &')

    device = torch.device('cpu')
    # device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')

    model = Net(n_hidden=1000, n_jnts=6).to(device=device)
    learning_rate = 1e-3
    batch_size = 64
    epochs = 200
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_data = IKDataSet('data_gen/cobotta_ik')
    test_data = IKDataSet('data_gen/cobotta_ik_test')
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    writer = SummaryWriter()
    global_step = [0]
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer, writer, global_step)
    model_path = 'tester/cobotta_model.pth'
    torch.save(model.state_dict(), model_path)
    print("Done!")
    writer.close()

    test_loop(test_dataloader, model, loss_fn)
