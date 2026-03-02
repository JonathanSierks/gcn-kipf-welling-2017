################# IMPORTS #################

from utils import *
from model import *

import argparse
import math
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

def train(model, optimizer, loss_function, epochs, x, y, A_hat, train_mask, val_mask, early_stop):
    patience = early_stop
    patience_left = patience
    best_loss = math.inf
    best_state = None


    train_loss, val_loss, val_acc = [], [], []
    best_val_loss, best_val_acc = 0, 0

    for epoch in range(epochs):

        # train block
        model.train()
        optimizer.zero_grad()
        output_train = model(x, A_hat)
        loss_train = loss_function(output_train[train_mask], y[train_mask])
        loss_train.backward()
        optimizer.step()

        # validation block
        model.eval()
        with torch.no_grad():
            output_val = model(x, A_hat)
            loss_val = loss_function(output_val[val_mask], y[val_mask])
            pred = output_val.argmax(dim=1)
            correct_val = (pred[val_mask] == y[val_mask]).sum().item()
            acc_val = correct_val/val_mask.sum().item()

        # early stopping
        if loss_val.item() < best_loss - 1e-6:
            best_loss = loss_val.item()
            patience_left = patience
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            # keep track of best validation loss & acc
            best_val_loss, best_val_acc = loss_val.item(), acc_val
        else:
            patience_left -= 1
            if patience_left == 0:
                print(f"Early stopping at epoch {epoch} applied. Best val loss: {best_loss:.4f} in epoch: {epoch-patience}")
                break

        # logging
        train_loss.append(loss_train.item())
        val_loss.append(loss_val.item())
        val_acc.append(acc_val)
        print(f"Epoch: {epoch+1} | Train Loss: {loss_train.item():.4f} | Val Loss: {loss_val.item():.4f} | Val Acc: {acc_val:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return train_loss, val_loss, val_acc, best_val_loss, best_val_acc

def test(model, x, y, A_hat, test_mask):
    model.eval()
    with torch.no_grad():
        out_test = model(x, A_hat)
        pred = out_test.argmax(dim=1)
        correct_test = (pred[test_mask] == y[test_mask]).sum().item()
        acc_test = correct_test / test_mask.sum().item()

    return acc_test

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="default_config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seeds = [int(s) for s in cfg["seeds"]]
    
    # device
    device = torch.device(cfg["train"]["device"] if torch.cuda.is_available() else "cpu")

    # download or (if already downloaded) reload data
    data = load_data(cfg["data"]["data_name"], os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"), cfg["data"]["data_class"])
    data_x = data.x.to(device)
    data_y = data.y.to(device)

    # hyperparameters
    lr = float(cfg["train"]["lr"])
    epochs = int(cfg["train"]["epochs"])
    weight_decay = float(cfg["train"]["weight_decay"])
    hidden_dim = int(cfg["model"]["hidden_dim"])
    dropout = float(cfg["model"]["dropout"])
    early_stopping = int(cfg["model"]["early_stopping"])

    input_dim = data.num_features
    output_dim = data.y.unique().size(0)

    # output set-up
    print("device:", device)
    print("HYPERPARAMS", "lr:", lr, "epochs:", epochs, "weight-decay:", weight_decay)
    print("MODEL ARCHITECTURE:", "input_dim:", input_dim, "hidden_dim:", hidden_dim, "output_dim:", output_dim, "dropout:", dropout)

    # prep data
    A_hat = compute_A_hat(data).to(device)
    train_mask = data.train_mask
    val_mask = data.val_mask
    test_mask = data.test_mask

    # results = {"train_loss": [], "val_acc": [], "test_acc": []}
    test_acc_history = []

    for i in range(len(seeds)):
            
        # set-up architecture
        model = GCN(input_dim, hidden_dim, output_dim,dropout).to(device)
        optimizer = optim.Adam(
            [
                {"params": model.layer1.parameters(), "weight_decay": weight_decay},    # as in kipf & welling: apply l2 reg only to the first layer
                {"params": model.layer2.parameters(), "weight_decay": 0.0},
            ],
            lr=lr)
        loss_function = nn.CrossEntropyLoss()

        # training (incl. validation assessments) & test accuracy
        train_loss, val_loss, val_acc, best_val_loss, best_val_acc = train(model, optimizer, 
                                                                        loss_function, epochs, 
                                                                        data_x, data_y, 
                                                                        A_hat, train_mask, 
                                                                        val_mask, early_stopping)
        test_acc = test(model, data_x, data_y, A_hat, test_mask)
        
        # output final results
        print(f" (Data Set: {cfg["data"]["data_name"]} in default config) Training Complete! | Test Accuracy: {test_acc} | Best Validation Loss: {best_val_loss}| Best Validation Accuracy: {best_val_acc}")
        test_acc_history.append(test_acc)
    
    print(f"Mean test accuracy of {np.mean(test_acc_history)} over {len(seeds)+1} runs with SD of {np.std(test_acc_history)}")

    # this is only needed if multiple experiments are to be carried out
    #results["train_loss"].append(train_loss)
    #results["val_acc"].append(val_acc)
    #results["test_acc"].append(test_acc)

if __name__ == "__main__":
    main()

# TO-DO
# - notebook aufbereiten: plotten etc.
# - notebook: verschiedene experimente?
# - nochmal ins paper schauen und hyperparams etc. kontrollieren
# - configs für die anderen datensets anlegen; ggf. alles über schleife laufen lassen und vollständige results in file schreiben?
#   - + was haben wir noch im DL kurs gemacht? ins assignment gucken; von da auch gut theorie übernehmbar
# - wie paper erweitern? was könnte ich noch testen?
# - aufschrieb kladde anfangen; theorie gur durchziehen