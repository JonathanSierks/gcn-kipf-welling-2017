################# IMPORTS #################

from utils import *
from model import *

import torch
print(torch.__version__)
print(torch.version.cuda)

'''
# install torch_geometric in-line since pip installation from requirements.txt is buggy
try:
    import torch_geometric
except ModuleNotFoundError:
    download_torch_geometrics()
    import torch_geometric
'''

# download or (if already downloaded) reload datasets
data_cora = load_planetoid("Cora", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
data_citeseer = load_planetoid("Citeseer", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
data_pubmed = load_planetoid("PubMed", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))

# compute A_hat
A_hat_cora = compute_A_hat(data_cora)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

layer = GCNLayer(data_cora.x.size()[1], data_cora.y.unique().size()[0])

Z = layer(data_cora.x, A_hat_cora)
print(Z.shape)

# data splitting/masking; device sending

# define GCN architecture --> models.py (there: GCN class etc.) & put everything together
# define BASELINE architecture --> baseline.py

# train and online evaluation GCN --> here
# train and online evaluation BASELINE --> here
# print progress & save to results




