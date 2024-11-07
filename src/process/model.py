from http.client import UnimplementedFileMode
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.conv import GatedGraphConv

torch.manual_seed(2020)


def get_conv_mp_out_size(in_size, last_layer, mps):
    size = in_size

    for mp in mps:
        size = round((size - mp["kernel_size"]) / mp["stride"] + 1)

    size = size + 1 if size % 2 != 0 else size

    return int(size * last_layer["out_channels"])


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv1d:
        torch.nn.init.xavier_uniform_(m.weight)

class Readout(nn.Module):
    def __init__(self, max_nodes, hidden_size, embed_size):
        super(Readout, self).__init__()
        self.max_nodes = max_nodes
        self.conv_y1 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3)
        self.pool_y1 = nn.MaxPool1d(kernel_size=3, stride=2)
        self.conv_y2 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=1)
        self.pool_y2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc_y = nn.Linear(hidden_size, 1)

        self.conv_z1 = nn.Conv1d(in_channels=hidden_size + embed_size, out_channels=hidden_size + embed_size, kernel_size=3)
        self.pool_z1 = nn.MaxPool1d(kernel_size=3, stride=2)
        self.conv_z2 = nn.Conv1d(in_channels=hidden_size + embed_size, out_channels=hidden_size + embed_size, kernel_size=1)
        self.pool_z2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc_z = nn.Linear(hidden_size + embed_size, 1)

        init_weights(self.conv_y1)
        init_weights(self.conv_y2)
        init_weights(self.conv_z1)
        init_weights(self.conv_z2)
        init_weights(self.fc_y)
        init_weights(self.fc_z)

        self.sigmoid = nn.Sigmoid()

    def forward(self, h, x):
        # TODO
        h = h.reshape(-1, self.max_nodes, h.shape[-1])
        x = x.reshape(-1, self.max_nodes, x.shape[-1])
        combined = torch.cat((h, x), dim=-1)
        y1 = self.pool_y1(F.relu(self.conv_y1(h.transpose(-2, -1))))
        y2 = self.pool_y2(F.relu(self.conv_y2(y1))).transpose(-2, -1)
        y = self.fc_y(y2)

        z1 = self.pool_z1(F.relu(self.conv_z1(combined.transpose(-2, -1))))
        z2 = self.pool_z2(F.relu(self.conv_z2(z1))).transpose(-2, -1)
        z = self.fc_z(z2)

        mult = torch.mul(y, z)
        avg = mult.mean(dim=1).squeeze(-1)
        return self.sigmoid(avg)

        
         

class Net(nn.Module):
    def __init__(self, gated_graph_conv_args, emb_size, max_nodes, device):
        super(Net, self).__init__()
        self.ggc = GatedGraphConv(**gated_graph_conv_args).to(device) 
        self.linear1 = nn.Linear(769, emb_size).to(device)
        init_weights(self.linear1)
        self.emb_size=emb_size
        self.readout = Readout(max_nodes, gated_graph_conv_args['out_channels'], emb_size).to(device)
        

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.linear1(x))
        h = self.ggc(x, edge_index)
        x = self.readout(h, x)

        return x

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
