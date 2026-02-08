import torch
import torch.nn as nn


class GCNLayer(nn.Module):
    def __init__(self, n_input, n_output ):
        super().__init__()
        self.W = nn.Parameter(torch.empty(n_input, n_output))
        nn.init.kaiming_normal_(self.W, mode="fan_in", nonlinearity="relu")

    def forward(self, X, A_hat):
        return torch.spmm(A_hat, (X @ self.W))
    
    