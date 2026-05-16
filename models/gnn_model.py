import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()

        # Graph convolution layers
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        # Final classifier
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index):

        # Layer 1
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        # Layer 2
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # Global representation (mean pooling)
        x = torch.mean(x, dim=0)

        # Output layer
        x = self.fc(x)

        return x

