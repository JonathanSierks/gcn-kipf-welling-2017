################# IMPORTS #################

from utils import *
from model import *

import argparse
import math
import torch.optim as optim
import torch
import time
import numpy as np
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
    best_epoch = -1
    best_val_acc = 0.0

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "epoch_train_time": [],
        "epoch_total_time": []
    }

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_start = time.perf_counter()

    for epoch in range(epochs):
        
        # account for timing missmatch when training on cuda
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        epoch_start = time.perf_counter()

        # train block
        model.train()
        optimizer.zero_grad()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        train_start = time.perf_counter()

        output_train = model(x, A_hat)
        loss_train = loss_function(output_train[train_mask], y[train_mask])
        loss_train.backward()
        optimizer.step()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        train_time = time.perf_counter() - train_start

        # validation block
        model.eval()
        with torch.no_grad():
            output_val = model(x, A_hat)
            loss_val = loss_function(output_val[val_mask], y[val_mask])
            pred = output_val.argmax(dim=1)
            correct_val = (pred[val_mask] == y[val_mask]).sum().item()
            acc_val = correct_val/val_mask.sum().item()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        epoch_total_time = time.perf_counter() - epoch_start

        # logging
        history["train_loss"].append(loss_train.item())
        history["val_loss"].append(loss_val.item())
        history["val_acc"].append(acc_val)
        history["epoch_train_time"].append(train_time)
        history["epoch_total_time"].append(epoch_total_time)

        # early stopping
        if loss_val.item() < best_loss - 1e-6:
            best_loss = loss_val.item()
            best_epoch = epoch
            best_val_acc = acc_val
            patience_left = patience
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        else:
            patience_left -= 1
            if patience_left == 0:
                print(f"Early stopping at epoch {epoch} applied. Best val loss: {best_loss:.4f} at epoch: {best_epoch}")
                break

        # output progress on individual epoch level
        print(
        f"Epoch: {epoch+1} | "
        f"Train Loss: {loss_train.item():.4f} | "
        f"Val Loss: {loss_val.item():.4f} | "
        f"Val Acc: {acc_val:.4f} | "
        f"Train Time: {train_time:.4f}s"
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_train_time = time.perf_counter() - total_start

    if best_state is not None:
        model.load_state_dict(best_state)

    summary = {
        "best_val_loss": best_loss,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "epochs_ran": len(history["train_loss"]),
        "total_train_time": total_train_time,
        "mean_epoch_train_time": float(np.mean(history["epoch_train_time"])),
        "mean_epoch_total_time": float(np.mean(history["epoch_total_time"])),
    }

    return history, summary

def test(model, x, y, A_hat, test_mask):
    model.eval()
    with torch.no_grad():
        out_test = model(x, A_hat)
        pred = out_test.argmax(dim=1)
        correct_test = (pred[test_mask] == y[test_mask]).sum().item()
        acc_test = correct_test / test_mask.sum().item()

    return acc_test

def main():
    all_histories = []
    run_summaries = []
    
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

    for i in range(len(seeds)):
        set_seed(seeds[i])  
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
        history, train_summary = train(model, optimizer, loss_function, epochs, data_x, data_y, A_hat, train_mask, val_mask, early_stopping)
        test_acc = test(model, data_x, data_y, A_hat, test_mask)
        
        run_summary = {
            "seed": seeds[i],
            **train_summary, 
            "test_acc": test_acc
        }
        
        all_histories.append(history)
        run_summaries.append(run_summary)

        # output final results
        print(
            f"(Data Set: {cfg['data']['data_name']} in default config) Training Complete! | "
            f"Test Accuracy: {run_summary['test_acc']:.4f} | "
            f"Best Validation Loss: {run_summary['best_val_loss']:.4f} | "
            f"Best Validation Accuracy: {run_summary['best_val_acc']:.4f}"
            f"Mean epoch total train time: {run_summary['mean_epoch_total_time']:.6f}"
        )
    
    print(
        f"Mean test accuracy of {np.mean([run['test_acc'] for run in run_summaries]):.4f} "
        f"over {len(seeds)} runs with SD of {np.std([run['test_acc'] for run in run_summaries], ddof=1):.4f}"
    )

    results = {
        "run_summaries": run_summaries,
        "all_histories": all_histories,
    }

    save_results(results, "results/gcn_results.json")

if __name__ == "__main__":
    main()

# TO-DO
# - notebook aufbereiten: plotten etc.
# - notebook: verschiedene experimente --> e.g. nochmal zeigen das mit etwas mehr hidden size auch acc noch weiter steigt?
# - nochmal ins paper schauen und hyperparams etc. kontrollieren

# t-SNE abbildung der finalen embeddings ausgeben und speichern

#   - + was haben wir noch im DL kurs gemacht? ins assignment gucken; von da auch gut theorie übernehmbar
# - wie paper erweitern? was könnte ich noch testen?
#   --> mini batch learning; dafür müsste ich dann aber am besten zeigen (nach kurzer überlegung), dass acc ungefähr gleich bleibt, die compute zeit aber runtergeht; dafür wiederum muss ich einen compute zeit tracker einbauen, so wie das auch kipf und welling gemacht haben
# - aufschrieb kladde anfangen; theorie gur durchziehen

# noch einen part adden, ggf. neues repo: graph classification (also die restlichen aufgaben aus dem assignment)