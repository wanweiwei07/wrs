import numpy as np
import torch.nn as nn
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation

def gen_data(robot, granularity, save_name='ik_data.csv'):
    data_set = []
    n_data_vec = np.ceil((robot.jnt_ranges[:, 1]-robot.jnt_ranges[:, 0])/granularity)
    n_data = np.prod(n_data_vec).astype(int)
    print(robot.jnt_ranges, n_data_vec)
    for i in tqdm(range(n_data)):
        jv_data = robot.rand_conf()
        tgt_pos, tgt_rotmat = robot.fk(jnt_values=jv_data, toggle_jacobian=False, update=False)
        tgt_wvec = Rotation.from_matrix(tgt_rotmat).as_rotvec()
        tcp_data = np.hstack((tgt_pos, tgt_wvec))
        data_set.append([jv_data, tcp_data])
    np.save(save_name, np.asarray(data_set))


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
        return xyzwvec, jnt_values


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

def my_collate_fn(batch):
    # Assume each element of 'batch' is a tuple (xyzwvec, jnt_values)
    # Unzip the batch
    xyzwvec_list, jnt_values_list = zip(*batch)

    # Convert lists to tensors
    xyzwvec_batch = torch.stack([torch.tensor(x, dtype=torch.float32) for x in xyzwvec_list])
    jnt_values_batch = torch.stack([torch.tensor(j, dtype=torch.float32) for j in jnt_values_list])

    return xyzwvec_batch, jnt_values_batch


if __name__ == '__main__':
    import os
    from datetime import datetime
    from torch.utils.data import DataLoader
    from wrs import robot_sim as rrcc
    from torch.utils.tensorboard import SummaryWriter
    import subprocess
    import webbrowser

    tb_command = "tensorboard --logdir=runs"
    proc = subprocess.Popen(tb_command, shell=True)
    webbrowser.open_new("http://localhost:6006/")

    robot = rrcc.Cobotta()
    granularity = np.pi*.3

    folder_path = robot.name+"_folder"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder created: {folder_path}")
    else:
        print(f"Folder already exists: {folder_path}")
    device = torch.device('cpu')
    model = Net(n_hidden=128, n_jnts=6).to(device=device)
    # check existance
    model_file = os.path.join(folder_path, "model.pth")
    if os.path.exists(model_file):
        print("Model exists, continued...")
        # Open and read the file
        with open(model_file, 'r') as file:
            model.load_state_dict(torch.load(model_file, map_location=device))
    learning_rate = 1e-2
    batch_size = 64
    epochs = 10
    loss_fn = nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    writer = SummaryWriter()
    global_step = [0]
    while True:
        file_name = datetime.now().strftime("%y%m%d%H%M%S")+"ik_data.csv"
        # Generate data
        gen_data(robot, granularity, save_name=os.path.join(folder_path, file_name))
        # Create dataloader
        train_dataloader = DataLoader(IKDataSet(os.path.join(folder_path, file_name)),
                                      batch_size=batch_size, shuffle=True,
                                      collate_fn=my_collate_fn)
        for t in range(epochs):
            for inputs, targets in train_dataloader:
                # Compute prediction and loss
                pred = model(inputs)
                loss = loss_fn(pred, targets)
                # print(loss.item())
                result_jv = pred.data.numpy()[0]
                random_jv = targets.data.numpy()[0]
                # print("jnt difference ", random_jv, result_jv, result_jv-random_jv, (result_jv-random_jv).T@(result_jv-random_jv), np.mean((result_jv-random_jv)**2))
                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                writer.add_scalar('loss', loss.item(), global_step[0])
                global_step[0] += 1
                print(f"Epoch {t + 1}, Loss: {loss.item()}")
        torch.save(model.state_dict(), model_file)
    writer.close()
