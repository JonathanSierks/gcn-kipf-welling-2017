import torch
import os
import numpy as np
import scipy.sparse

from scipy.sparse import coo_matrix, diags
from torch_geometric.datasets import Planetoid

def download_torch_geometrics():

    print(torch.__version__)
    print(torch.version.cuda)
    print(torch.cuda.is_available())

    import torch

    TORCH = torch.__version__.split('+')[0]

    if torch.cuda.is_available():
        CUDA = 'cu' + torch.version.cuda.replace('.', '')
    else:
        CUDA = 'cpu'

    !pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv \
        -f https://data.pyg.org/whl/torch-{TORCH}+{CUDA}.html

    !pip install torch-geometric

def load_planetoid(name, data_root):

    dataset_path = os.path.join(data_root, "Planetoid", name)

    if not os.path.exists(dataset_path):
        print(f"Dataset {name} not yet available. Download starts ...")
    else:
        print(f"Dataset {name} already available, no download necessary.")

    dataset = Planetoid(root=os.path.join(data_root, "Planetoid"), name=name)

    return dataset

def compute_A_hat(data):

    # print data stats
    print("------------- DATA SET INFOS ------------- ")
    print(f"Number of nodes: {data.num_nodes}")
    print(f"Number of edges: {data.num_edges}")
    print(f"Number of features per Node: {data.x.size()[1]}")
    print(f"Number of targets in this dataset: {data.y.unique().size()[0]}")

    rows = data.edge_index[0]
    cols = data.edge_index[1]
    values = np.ones(len(rows))

    A = scipy.sparse.coo_matrix((values, (rows, cols)), shape=(data.num_nodes, data.num_nodes), dtype=np.float32)
    degrees = A.sum(axis=1).A1
    D = scipy.sparse.diags(degrees, dtype=np.float32)

    A_self = A + scipy.sparse.diags(np.ones(data.num_nodes), dtype=np.float32)

    degrees = A_self.sum(axis=1).A1
    D_norm = scipy.sparse.diags(degrees, dtype=np.float32)

    A_hat = (D_norm.power(-0.5) @ A_self @ D_norm.power(-0.5)).tocoo()

    A_processed = torch.sparse_coo_tensor(A_hat.nonzero(), 
                        A_hat.data, 
                        size=A_hat.shape, 
                        dtype=torch.float32)
    return A_processed