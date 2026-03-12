################# IMPORTS #################

from utils import *
from model import *
from typing import Any
from dataclasses import dataclass

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

@dataclass
class TrainConfig:
    epochs: int
    early_stop: int
    batch_mode: str
    batch_size: int
    num_neighbors: list
    device: Any

@dataclass
class GraphDataBundle:
    x: torch.Tensor
    y: torch.Tensor
    A_hat: torch.Tensor
    train_mask: torch.Tensor
    val_mask: torch.Tensor
    test_mask: torch.Tensor
    data: Any = None    

# wrapper for the 2 train methods (full vs. mini-batch)
def train(model, optimizer, loss_function, graph_data, train_cfg, train_loader, val_loader):
    if train_cfg.batch_mode == "full_batch":
        return train_full_batch(model, optimizer, loss_function, graph_data, train_cfg)
    elif train_cfg.batch_mode == "mini_batch":
        return train_mini_batch(model, optimizer, loss_function, train_cfg, train_loader, val_loader)
    else:
        raise ValueError(f"Unknown batch_mode parameter: {train_cfg.batch_mode}")

def train_mini_batch(model, optimizer, loss_function, train_cfg, train_loader, val_loader):
    
    # ----------- set up -----------
    epochs = train_cfg.epochs
    early_stop = train_cfg.early_stop
    device = train_cfg.device
    
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
        "epoch_total_time": [],
    }

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_start = time.perf_counter()

    for epoch in range(epochs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        epoch_start = time.perf_counter()

        # ----------- train -----------
        model.train()
        batch_losses = []

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        train_start = time.perf_counter()

        # as we now have batches we iteratre over every subgraph in the batch 
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            out = model(batch.x, batch.edge_index)

            out_target = out[:batch.batch_size]
            y_target = batch.y[:batch.batch_size]

            loss = loss_function(out_target, y_target)
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        train_time = time.perf_counter() - train_start

        train_loss_epoch = float(np.mean(batch_losses))

        # ----------- evaluation -----------
        model.eval()

        val_losses = []
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)

                out = model(batch.x, batch.edge_index)
                out_target = out[:batch.batch_size]
                y_target = batch.y[:batch.batch_size]

                loss = loss_function(out_target, y_target)
                val_losses.append(loss.item())

                pred = out_target.argmax(dim=1)
                correct += (pred == y_target).sum().item()
                total += y_target.size(0)

        val_loss_epoch = float(np.mean(val_losses))
        val_acc_epoch = correct / total

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        epoch_total_time = time.perf_counter() - epoch_start

        # ----------- logging per epoch -----------
        history["train_loss"].append(train_loss_epoch)
        history["val_loss"].append(val_loss_epoch)
        history["val_acc"].append(val_acc_epoch)
        history["epoch_train_time"].append(train_time)
        history["epoch_total_time"].append(epoch_total_time)

        # ----------- early stopping -----------
        if val_loss_epoch < best_loss - 1e-6:
            best_loss = val_loss_epoch
            best_val_acc = val_acc_epoch
            best_epoch = epoch
            patience_left = patience
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_left -= 1
            if patience_left == 0:
                print(
                    f"Early stopping at epoch {epoch+1} applied. "
                    f"Best val loss: {best_loss:.4f} at epoch {best_epoch+1}"
                )
                break

        print(
            f"Epoch: {epoch+1} | "
            f"Train Loss: {train_loss_epoch:.4f} | "
            f"Val Loss: {val_loss_epoch:.4f} | "
            f"Val Acc: {val_acc_epoch:.4f} | "
            f"Train Time: {train_time:.4f}s"
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_train_time = time.perf_counter() - total_start

    if best_state is not None:
        model.load_state_dict(best_state)

    # ----------- logging per run -----------
    summary = {
        "best_val_loss": best_loss,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch + 1,
        "epochs_ran": len(history["train_loss"]),
        "total_train_time": total_train_time,
        "mean_epoch_train_time": float(np.mean(history["epoch_train_time"])),
        "mean_epoch_total_time": float(np.mean(history["epoch_total_time"])),
    }

    return history, summary

def train_full_batch(model, optimizer, loss_function, graph_data, train_cfg):
    
    # ----------- set up -----------
    patience = train_cfg.early_stop
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

    for epoch in range(train_cfg.epochs):
        
        # account for timing missmatch when training on cuda
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        epoch_start = time.perf_counter()

        # ----------- train -----------
        model.train()
        optimizer.zero_grad()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        train_start = time.perf_counter()

        output_train = model(graph_data.x, graph_data.A_hat)
        loss_train = loss_function(output_train[graph_data.train_mask], graph_data.y[graph_data.train_mask])
        loss_train.backward()
        optimizer.step()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        train_time = time.perf_counter() - train_start

        # ----------- validation -----------
        model.eval()
        with torch.no_grad():
            output_val = model(graph_data.x, graph_data.A_hat)
            loss_val = loss_function(output_val[graph_data.val_mask], graph_data.y[graph_data.val_mask])
            pred = output_val.argmax(dim=1)
            correct_val = (pred[graph_data.val_mask] == graph_data.y[graph_data.val_mask]).sum().item()
            acc_val = correct_val/graph_data.val_mask.sum().item()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        epoch_total_time = time.perf_counter() - epoch_start

        # ----------- logging per epoch -----------
        history["train_loss"].append(loss_train.item())
        history["val_loss"].append(loss_val.item())
        history["val_acc"].append(acc_val)
        history["epoch_train_time"].append(train_time)
        history["epoch_total_time"].append(epoch_total_time)

        # ----------- early stopping -----------
        if loss_val.item() < best_loss - 1e-6:
            best_loss = loss_val.item()
            best_epoch = epoch
            best_val_acc = acc_val
            patience_left = patience
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        else:
            patience_left -= 1
            if patience_left == 0:
                print(f"Early stopping at epoch {epoch+1} applied. Best val loss: {best_loss:.4f} at epoch: {best_epoch+1}")
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

    # ----------- logging per run -----------
    summary = {
        "best_val_loss": best_loss,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch + 1,
        "epochs_ran": len(history["train_loss"]),
        "total_train_time": total_train_time,
        "mean_epoch_train_time": float(np.mean(history["epoch_train_time"])),
        "mean_epoch_total_time": float(np.mean(history["epoch_total_time"])),
    }

    return history, summary

# wrapper for the 2 test methods (full vs. mini-batch)
def test(model, train_cfg,  graph_data, test_loader):
    if train_cfg.batch_mode == "full_batch":
        return test_full_batch(model, graph_data)
    elif train_cfg.batch_mode == "mini_batch":
        return test_mini_batch(model, test_loader, train_cfg.device)
    else:
        raise ValueError(f"Unknown batch_mode parameter: {train_cfg.batch_mode}")
    
def test_full_batch(model, graph_data):
    model.eval()
    with torch.no_grad():
        out_test = model(graph_data.x, graph_data.A_hat)
        pred = out_test.argmax(dim=1)
        correct_test = (pred[graph_data.test_mask] == graph_data.y[graph_data.test_mask]).sum().item()
        acc_test = correct_test / graph_data.test_mask.sum().item()

    return acc_test

def test_mini_batch(model, test_loader, device):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)

            out = model(batch.x, batch.edge_index)
            out_target = out[:batch.batch_size]
            y_target = batch.y[:batch.batch_size]

            pred = out_target.argmax(dim=1)
            correct += (pred == y_target).sum().item()
            total += y_target.size(0)

    return correct / total

def run_from_cfg(cfg):
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
    num_neighbors = cfg["data"]["num_neighbors"]
    batch_mode = cfg["data"]["batch_mode"]
    batch_size = int(cfg["data"]["batch_size"])

    input_dim = data.num_features
    output_dim = data.y.unique().size(0)

    # output set-up
    print("device:", device)
    print("HYPERPARAMS", "lr:", lr, "epochs:", epochs, "weight-decay:", weight_decay)
    print("MODEL ARCHITECTURE:", "input_dim:", input_dim, "hidden_dim:", hidden_dim, "output_dim:", output_dim, "dropout:", dropout)

    # prep data
    #A_hat = compute_A_hat(data).to(device)
    train_mask = data.train_mask
    val_mask = data.val_mask
    test_mask = data.test_mask

    # wrap graph and train parameters in dataclasses
    train_cfg = TrainConfig(
        epochs=epochs,
        early_stop=early_stopping,
        batch_mode=batch_mode,
        batch_size=batch_size,
        num_neighbors=num_neighbors,
        device=device
    )

    graph_data = GraphDataBundle(
        x=data_x,
        y=data_y,
        A_hat= None,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        data=data
    )

    all_histories = []
    run_summaries = []

    for i in range(len(seeds)):
        set_seed(seeds[i])  

        # set-up architecture
        # decide here if we use full or minibatch based approach
        train_loader, val_loader, test_loader = None, None, None
        if batch_mode == "full_batch":
            A_hat = compute_A_hat(data).to(device)
            graph_data.A_hat = A_hat
            model = GCNFullBatch(input_dim, hidden_dim, output_dim, dropout).to(device)

        elif batch_mode == "mini_batch":
            train_loader, val_loader, test_loader = build_neighbor_loaders(graph_data, train_cfg)
            model = GCNMiniBatch(input_dim, hidden_dim, output_dim, dropout).to(device)

        else:
            raise ValueError(f"Unknown batch_mode: {batch_mode}")

        optimizer = optim.Adam(
            [
                {"params": model.layer1.parameters(), "weight_decay": weight_decay},    # as in kipf & welling: apply l2 reg only to the first layer
                {"params": model.layer2.parameters(), "weight_decay": 0.0},
            ],
            lr=lr)
        loss_function = nn.CrossEntropyLoss()

        # training (incl. validation assessments) & test run
        history, train_summary = train(model, optimizer, loss_function, graph_data, train_cfg, train_loader, val_loader)
        test_acc = test(model, train_cfg, graph_data, test_loader)
        
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
            f"Best Validation Accuracy: {run_summary['best_val_acc']:.4f} | "
            f"Mean epoch total train time: {run_summary['mean_epoch_total_time']:.6f}s"
        )
    
    print(
        f"Mean test accuracy of {np.mean([run['test_acc'] for run in run_summaries]):.4f} "
        f"over {len(seeds)} runs with SD of {np.std([run['test_acc'] for run in run_summaries], ddof=1):.4f}"
    )

    results = {
        "run_summaries": run_summaries,
        "all_histories": all_histories,
    }

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="default_config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    results = run_from_cfg(cfg)

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