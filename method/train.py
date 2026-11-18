"""Train convNetLeonArDBO (from models.py) for 1D, 2D or 3D grids.

Replaces the training code that was duplicated across neural_networks.py (2D),
neural_networks1.py (1D) and neural_networks3.py (3D).

Example:
    python train.py --dim 2
    python train.py --dim 3 --epochs 2000 --loss kl --comet
"""
import argparse
import os

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from models import convNetLeonArDBO
from utils import CustomDataset, get_the_best_past_y_and_x


class likelihoodHellingerLoss(nn.Module):

    def forward(self, pred_posterior, posterior):
        pred_posterior = torch.exp(pred_posterior) / torch.sum(torch.exp(pred_posterior), dim=1).unsqueeze(1)
        posterior = torch.exp(posterior) / torch.sum(torch.exp(posterior), dim=1).unsqueeze(1)
        return torch.mean(0.5 * torch.sum((torch.sqrt(pred_posterior + 1e-15) - torch.sqrt(posterior + 1e-15)) ** 2, dim=1))


def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for priors, posteriors, new_xs, new_ys, y_stars in train_loader:
        priors, posteriors, new_xs, new_ys, y_stars = priors.to(device), posteriors.to(device), new_xs.to(device), new_ys.to(device), y_stars.to(device)
        optimizer.zero_grad()
        outputs = model(new_ys, y_stars, new_xs, priors)
        loss = criterion(outputs, posteriors)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def validate(model, test_loader, criterion, other_criterion, device):
    model.eval()
    total_loss = 0
    total_other_loss = 0
    with torch.no_grad():
        for priors, posteriors, new_xs, new_ys, y_stars in test_loader:
            priors, posteriors, new_xs, new_ys, y_stars = priors.to(device), posteriors.to(device), new_xs.to(device), new_ys.to(device), y_stars.to(device)
            outputs = model(new_ys, y_stars, new_xs, priors)
            total_loss += criterion(outputs, posteriors).item()
            total_other_loss += other_criterion(outputs, posteriors).item()
    return total_loss / len(test_loader), total_other_loss / len(test_loader)


def load_data(data_dir, data_name, dim, grid_size, n_runs, horizon, initial_value, train_ratio):
    """Load priors/posteriors/xs/ys, preprocess, and split by run into train/test datasets."""
    def load(prefix):
        return torch.load(os.path.join(data_dir, f"{prefix}_{data_name}.pt"))

    priors = load("priors")
    posteriors = load("posteriors")
    new_xs = load("new_xs")
    new_ys = load("new_ys")

    # x in [0, 1] -> integer grid index
    new_xs = (new_xs * (grid_size - 1)).round().long()

    new_xs = new_xs.view(-1, dim)
    new_ys = new_ys.view(-1, 1)
    priors = torch.log(priors.view(-1, *[grid_size] * dim) + 1e-15)
    posteriors = (posteriors + 1e-15).view(n_runs * horizon, -1)

    past_best_y, _, _, _ = get_the_best_past_y_and_x(new_ys, new_xs, horizon, n_runs, initial_value, dim=dim, grid_size=grid_size)
    past_best_y = new_ys - past_best_y

    # split on a run boundary so no run is shared between train and test
    train_split = int(train_ratio * n_runs) * horizon

    train_dataset = CustomDataset(priors[:train_split], posteriors[:train_split], new_xs[:train_split],
                                  new_ys[:train_split], past_best_y[:train_split])
    test_dataset = CustomDataset(priors[train_split:], posteriors[train_split:], new_xs[train_split:],
                                 new_ys[train_split:], past_best_y[train_split:])
    print(f"train samples: {len(train_dataset)}, test samples: {len(test_dataset)}")
    return train_dataset, test_dataset


def train_validate_save(dim, data_name=None, data_dir=".", grid_size=11, n_runs=2000, horizon=30,
                        batch_size=512, num_epochs=2000, train_ratio=0.9, initial_value=-10,
                        device="cuda", lr=0.001, betas=(0.9, 0.999), training_loss="kl",
                        larger_decay=1.0, smaller_decay=1e-4, schedule=False, detect_anomaly=False,
                        best_model_path=None, final_model_path=None, experiment=None):

    if data_name is None:
        data_name = f"d1_g{grid_size}_N{n_runs}_T{horizon}_M500000_pthompson_dim{dim}_lowls"
    if best_model_path is None:
        best_model_path = f"lowls{dim}d.pth"
    if final_model_path is None:
        final_model_path = f"final_model_{dim}d_{training_loss}.pth"

    hyper_params = dict(locals())
    hyper_params.pop("experiment")
    print(hyper_params)
    if experiment is not None:
        experiment.log_parameters(hyper_params)

    torch.autograd.set_detect_anomaly(detect_anomaly)

    train_dataset, test_dataset = load_data(data_dir, data_name, dim, grid_size, n_runs, horizon,
                                            initial_value, train_ratio)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = convNetLeonArDBO(dim=dim, grid_size=grid_size).to(device)
    print("trainable params:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    criteria = {
        "cse": nn.CrossEntropyLoss(),
        "hel": likelihoodHellingerLoss(),
        "kl": nn.KLDivLoss(reduction="batchmean"),
    }
    other = {"cse": "kl", "hel": "cse", "kl": "cse"}
    criterion = criteria[training_loss]
    other_criterion = criteria[other[training_loss]]

    # heavier weight decay on the prior-encoder / posterior-decoder path
    optimizer = torch.optim.Adam([
        {"params": model.priorEncoder.parameters(), "weight_decay": larger_decay},
        {"params": model.posteriorDecoder.parameters(), "weight_decay": larger_decay},
        {"params": model.yEncoder.parameters(), "weight_decay": smaller_decay},
        {"params": model.ystarEncoder.parameters(), "weight_decay": smaller_decay},
        {"params": model.likelihoodDecoder.parameters(), "weight_decay": smaller_decay},
        {"params": model.activation.parameters(), "weight_decay": smaller_decay},
    ], lr=lr, betas=betas)

    scheduler = ReduceLROnPlateau(optimizer, "min", factor=0.5) if schedule else None

    best_val_loss = float("inf")
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss, other_val_loss = validate(model, test_loader, criterion, other_criterion, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, "
              f"Validation Loss: {val_loss:.4f}, Other Validation Loss: {other_val_loss:.4f}")

        if experiment is not None:
            experiment.log_metric(name="train_loss", value=train_loss, step=epoch)
            experiment.log_metric(name="validation_loss", value=val_loss, step=epoch)

        if scheduler is not None:
            scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model, best_model_path)
            print(f"Saved best model with validation loss: {val_loss:.4f}")

    torch.save(model, final_model_path)
    print(f"Saved final model after {num_epochs} epochs")
    return model


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dim", type=int, required=True, choices=[1, 2, 3])
    p.add_argument("--data_name", default=None, help="default: d1_g11_N2000_T30_M500000_pthompson_dim{dim}_lowls")
    p.add_argument("--data_dir", default=".")
    p.add_argument("--grid_size", type=int, default=11)
    p.add_argument("--n_runs", type=int, default=2000, help="number of BO runs (N) in the dataset")
    p.add_argument("--horizon", type=int, default=30, help="steps per run (T) in the dataset")
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--train_ratio", type=float, default=0.9)
    p.add_argument("--initial_value", type=float, default=-10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--loss", default="kl", choices=["kl", "cse", "hel"])
    p.add_argument("--larger_decay", type=float, default=1.0)
    p.add_argument("--smaller_decay", type=float, default=1e-4)
    p.add_argument("--schedule", action="store_true", help="use ReduceLROnPlateau")
    p.add_argument("--detect_anomaly", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--best_model_path", default=None, help="default: lowls{dim}d.pth")
    p.add_argument("--final_model_path", default=None)
    p.add_argument("--comet", action="store_true",
                   help="log to Comet (API key read from COMET_API_KEY env var / ~/.comet.config)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    experiment = None
    if args.comet:
        from comet_ml import start
        experiment = start(project_name="argmaxupdates", workspace="emkaragozlu")

    train_validate_save(
        dim=args.dim, data_name=args.data_name, data_dir=args.data_dir, grid_size=args.grid_size,
        n_runs=args.n_runs, horizon=args.horizon, batch_size=args.batch_size, num_epochs=args.epochs,
        train_ratio=args.train_ratio, initial_value=args.initial_value, device=args.device, lr=args.lr,
        training_loss=args.loss, larger_decay=args.larger_decay, smaller_decay=args.smaller_decay,
        schedule=args.schedule, detect_anomaly=args.detect_anomaly,
        best_model_path=args.best_model_path, final_model_path=args.final_model_path,
        experiment=experiment,
    )
