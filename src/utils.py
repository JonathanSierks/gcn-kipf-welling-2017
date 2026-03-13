import torch
import os
import numpy as np
import scipy.sparse
import yaml
import random
import json

from scipy.sparse import coo_matrix, diags
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import NeighborLoader

import os.path as osp
import pandas as pd
import torch
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset

'''
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
'''

def load_data(name, data_root, data_class):
    dataset_path = os.path.join(data_root, data_class, name)

    if os.path.exists(dataset_path):
        print(f"Dataset {name} already available, no download necessary.")
    else:
        print(f"Dataset {name} not yet available. Download starts ...")

    if name in ["Cora", "CiteSeer", "PubMed"]:
        data = Planetoid(
            root=os.path.join(data_root, data_class),
            name=name
        )[0]
        return data

    elif name == "ogbn-arxiv":
        root = dataset_path

        if not hasattr(torch, "_original_load"):
            torch._original_load = torch.load

        def patched_torch_load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return torch._original_load(*args, **kwargs)

        torch.load = patched_torch_load

        class PygOgbnArxiv(PygNodePropPredDataset):
            def __init__(self):
                master = pd.read_csv(osp.join(root, "ogbn-master.csv"), index_col=0)
                meta_dict = master[name].fillna("None").astype(str).to_dict()
                meta_dict["dir_path"] = root

                super().__init__(
                    name=name,
                    root=root,
                    transform=None,
                    meta_dict=meta_dict
                )

            def get_idx_split(self):
                split_type = self.meta_info["split"]
                path = osp.join(self.root, "split", split_type)

                train_idx = torch.tensor(pd.read_csv(osp.join(path, "train.csv.gz"), header=None).iloc[:, 0].values, dtype=torch.long)
                valid_idx = torch.tensor(pd.read_csv(osp.join(path, "valid.csv.gz"), header=None).iloc[:, 0].values, dtype=torch.long)
                test_idx  = torch.tensor(pd.read_csv(osp.join(path, "test.csv.gz"),  header=None).iloc[:, 0].values, dtype=torch.long)

                return {"train": train_idx, "valid": valid_idx, "test": test_idx}

        dataset = PygOgbnArxiv()
        data = dataset[0]
        split_idx = dataset.get_idx_split()

        n = data.num_nodes
        data.train_mask = torch.zeros(n, dtype=torch.bool)
        data.val_mask   = torch.zeros(n, dtype=torch.bool)
        data.test_mask  = torch.zeros(n, dtype=torch.bool)

        data.train_mask[split_idx["train"]] = True
        data.val_mask[split_idx["valid"]] = True
        data.test_mask[split_idx["test"]] = True

        data.y = data.y.view(-1)

        return data
    
    else:
        raise ValueError(f"Unknown dataset: {name}")


'''
def load_data(name, data_root, data_class):

    dataset_path = os.path.join(data_root, data_class, name)

    if not os.path.exists(dataset_path):
        print(f"Dataset {name} not yet available. Download starts ...")
    else:
        print(f"Dataset {name} already available, no download necessary.")

    dataset = Planetoid(root=os.path.join(data_root, data_class), name=name)[0]

    return dataset
'''
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
    
    A_self = A + scipy.sparse.diags(np.ones(data.num_nodes), dtype=np.float32)      # A-matrix + I_N-matrix

    degrees = A_self.sum(axis=1).A1        # D-matrix
    D_norm = scipy.sparse.diags(degrees, dtype=np.float32)

    A_hat = (D_norm.power(-0.5) @ A_self @ D_norm.power(-0.5)).tocoo()

    # transform A_hat to torch sparse tensor
    coo = A_hat.tocoo()
    indices = np.vstack((coo.row, coo.col))
    indices = torch.from_numpy(indices).long()
    values  = torch.from_numpy(coo.data).float()
    A_processed = torch.sparse_coo_tensor(indices, values, size=coo.shape).coalesce()
    
    return A_processed

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def save_results(results, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)

def build_neighbor_loaders(graph_data, train_cfg):
    data = graph_data.data
    num_neighbors = train_cfg.num_neighbors
    batch_size = train_cfg.batch_size

    train_loader = NeighborLoader(
        data,
        input_nodes=data.train_mask,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = NeighborLoader(
        data,
        input_nodes=data.val_mask,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        shuffle=False,
    )

    test_loader = NeighborLoader(
        data,
        input_nodes=data.test_mask,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader