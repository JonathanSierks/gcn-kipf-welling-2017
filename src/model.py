import torch
import torch.nn as nn


class GCNLayer(nn.Module):
    def __init__(self, n_input, n_output):
        super().__init__()
        self.W = nn.Parameter(torch.empty(n_input, n_output))
        nn.init.kaiming_normal_(self.W, mode="fan_in", nonlinearity="relu")

    def forward(self, X, A_hat):
        return torch.spmm(A_hat, (X @ self.W))

class GCN(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, dropout):
        super().__init__()
        self.layer1 = GCNLayer(n_input, n_hidden)
        self.layer2 = GCNLayer(n_hidden, n_output)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, X, A_hat):
        X = self.layer1(X, A_hat)
        X = self.relu(X)
        X = self.dropout(X)
        X = self.layer2(X, A_hat)
        return X
    