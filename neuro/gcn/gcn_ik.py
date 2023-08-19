import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset
import math
import itertools
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid


class IKDataSet(Dataset):
    def __init__(self, file, transform=None):
        # self.ik_frame = pd.read_csv(file)
        self.ik_tcp = np.load(file+"_tcp.npy")
        self.ik_jnts = np.load(file+"_jnts.npy")
        self.transform = transform
        # _min_max = np.load(file+"_min_max.npy")
        # self.min = _min_max[0]
        # self.max = _min_max[1]

    def __len__(self):
        return len(self.ik_tcp)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        tcp_values = self.ik_tcp[idx]
        # xyzrpy = (xyzrpy-self.min)/(self.max-self.min) #normalize
        jnt_values = self.ik_jnts[idx]
        return torch.Tensor(tcp_values), torch.Tensor(jnt_values)


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

if __name__ == '__main__':

    # Set up your training parameters
    batch_size = 3000
    num_epochs = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare your training data
    train_dataset = IKDataSet('cobotta_ik')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Set up your model and optimizer
    input_size = 1
    output_size = 1
    model = GCN(input_size, 32, output_size)
    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=5e-6)

    # Define your loss function
    criterion = nn.CrossEntropyLoss()
    a = [0,1,2,3,4,5]
    p = itertools.product(a,a)
    edge_index = torch.tensor(np.array(list(p)).T.tolist(), dtype=torch.long)

    # Train your model
    model.train()
    for epoch in range(num_epochs):
        for batch, (batch_inputs, batch_labels) in enumerate(train_dataloader):
            batch_inputs = batch_inputs
            batch_labels = batch_labels
            edge_index = edge_index
            # Clear gradients
            optimizer.zero_grad()

            # Forward pass
            print(batch_inputs.shape)
            outputs = model(batch_inputs, edge_index=edge_index)
            # Reshape output and target for loss calculation
            # outputs = outputs.view(-1, output_size)
            # batch_labels = batch_labels.view(-1)

            # Compute loss
            loss = criterion(outputs, batch_labels)

            # Backward pass
            loss.backward()

            # Clip gradients to avoid exploding gradients
            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

            # Update model parameters
            optimizer.step()

            # Print loss for tracking progress
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch loss: {loss.item()}")

    # Save your trained model
    torch.save(model.state_dict(), "trained_model.pth")
