import torch
import torch.nn as nn


class GCNLayer(nn.Module):
    def __init__(self, n_input, n_output ):
        super().__init__()
        self.W = nn.Parameter(torch.empty(n_input, n_output))
        nn.init.kaiming_normal_(self.W, mode="fan_in", nonlinearity="relu")

    def forward(self, X, A_hat):
        return torch.spmm(A_hat, (X @ self.W))

class GCN(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super().__init__()
        self.layer1 = GCNLayer(n_input, n_hidden)
        self.layer2 = GCNLayer(n_hidden, n_output)

    def forward(self, X, A_hat):
        xw_1 = self.layer1(X, A_hat)
        act_1 = torch.relu(xw_1)
        xw_2 = self.layer2(xw_1, A_hat)
        return xw_2
    