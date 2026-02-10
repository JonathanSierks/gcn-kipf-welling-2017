################# IMPORTS #################

from utils import *
from model import *

import torch.optim as optim
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

def train(model, optimizer, loss_function, epochs, x, y, A_hat, train_mask, val_mask):

    train_loss, val_acc = [], []
    
    for epoch in range(epochs):

        model.train()
        optimizer.zero_grad()
        output_train = model(x, A_hat)
        loss_train = loss_function(output_train[train_mask], y[train_mask])
        loss_train.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            output_val = model(x, A_hat)
            loss_val = loss_function(output_val[val_mask], y[val_mask])



# download or (if already downloaded) reload datasets
data_cora = load_planetoid("Cora", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
data_citeseer = load_planetoid("Citeseer", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
data_pubmed = load_planetoid("PubMed", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# set-up data stuff
data_cora_x = data_cora.x.to(device)
data_cora_y = data_cora.y.to(device)

A_hat_cora = compute_A_hat(data_cora).to_device(device)
train_mask = data_cora.train_mask
val_mask = data_cora.val_mask
test_mask = data_cora.test_mask


# set-up architecture
model = GCN(1433, 32,7,0).to_device(device)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
loss_function = nn.CrossEntropyLoss()


# training
train(model, optimizer, loss_function, epochs, data_x, data_y, A_hat)



model.eval()










gcn = GCN(2707, 32, 7)

# load hyperparams from config.yaml





# data splitting/masking; device sending

# define GCN architecture --> models.py (there: GCN class etc.) & put everything together
# define BASELINE architecture --> baseline.py

# train and online evaluation GCN --> here
# train and online evaluation BASELINE --> here
# print progress & save to results




