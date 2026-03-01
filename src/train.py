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

    train_loss, val_loss, val_acc = [], [], []
    
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
            pred = output_val.argmax(dim=1)
            correct_val = (pred[val_mask] == y[val_mask]).sum().item()
            acc_val = correct_val/val_mask.sum().item()

        train_loss.append(loss_train)
        val_loss.append(loss_val)
        val_acc.append(acc_val)

    return train_loss, val_loss, val_acc



def test():
    pass


# load hyperparams from config.yaml





# data splitting/masking; device sending

# define GCN architecture --> models.py (there: GCN class etc.) & put everything together
# define BASELINE architecture --> baseline.py

# train and online evaluation GCN --> here
# train and online evaluation BASELINE --> here
# print progress & save to results

def main():
    cfg = load_config("default.yaml")
    
    # device
    device = torch.device(cfg["train"]["device"] if torch.cuda.is_available() else "cpu")

    # hyperparameters
    lr = cfg["train"]["lr"]
    epochs = cfg["train"]["epochs"]
    hidden_dim = cfg["model"]["hidden_dim"]
    dropout = cfg["model"]["dropout"]
    weight_decay = cfg["model"]["weight_decay"]
    # ...

    print("device:", device)
    print("lr:", lr, "epochs:", epochs)

    # download or (if already downloaded) reload data and prep
    data = load_data("hier: keyword welches data set; e.g. PubMed (aus configs.yaml)", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
    data_x = data.x.to(device)
    data_y = data.y.to(device)

    A_hat = compute_A_hat(data).to_device(device)
    train_mask = data.train_mask
    val_mask = data.val_mask
    test_mask = data.test_mask

    results = {"train_loss": [], "val_acc": [], "test_acc": []}

    # set-up architecture
    model = GCN(1433, 32,7,0).to_device(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_function = nn.CrossEntropyLoss()

    # training; incl. validation output every x epochs
    train_loss, val_acc = train(model, optimizer, loss_function, epochs, data_x, data_y, A_hat)
    test_acc = test()

    results["train_loss"].append(train_loss)
    results["val_acc"].append(val_acc)
    results["test_acc"].append(test_acc)












    gcn = GCN(2707, 32, 7)



if __name__ == "__main__":
    main()
