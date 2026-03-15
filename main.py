"""
Main experiment runner for CPU Forecasting journal.

Runs the full 3-phase evaluation for each deep learning model:
  1. Train on Azure   -> Eval on Azure   -> Save results
  2. Fine-tune on Alibaba -> Eval on Alibaba -> Save results
  3. Eval on Azure (Post-Fine-tuning) -> Save results (Catastrophic Forgetting check)

Configurations tested: 6-1, 6-2, 6-3, 12-1, 12-2, 12-3, 24-1, 24-2, 24-3

Usage:
    python main.py --model all
    python main.py --model bilstm rnn lstm
    python main.py --model informer --azure_dir datasets/azure --num_epochs 30

Note: For ARIMA baseline, run arima/arima.py directly.
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, Dataset, Subset

# -------------------------------------------------------------------------
# MODEL IMPORTS
# -------------------------------------------------------------------------

import sys
sys.path.insert(0, os.path.dirname(__file__))

from bilstm.bilstm import BiLSTMModel
from informer.informer import InformerModel
from lstm.lstm import LSTMModel
from rnn.rnn import RNNModel
from tft.tft import TemporalFusionTransformer

# -------------------------------------------------------------------------
# SHARED DATASET
# -------------------------------------------------------------------------

class VMCPUDataset(Dataset):
    """
    Loads per-VM CPU time series and applies per-VM min-max scaling
    fit exclusively on the training split.
    Returns sequences of shape [seq_len, num_vms, 1].
    """

    def __init__(self, csv_dir, seq_len=6, pred_len=1, train_end_idx=None):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.data_per_vm = []

        csv_files = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {csv_dir}")

        print(f"Loading {len(csv_files)} CSV files from {csv_dir}...")

        for csv_path in csv_files:
            df = pd.read_csv(csv_path)
            col_name = 'avg cpu' if 'avg cpu' in df.columns else df.columns[0]
            cpu_values = df[col_name].astype('float32').values

            train_slice = cpu_values if train_end_idx is None else cpu_values[:train_end_idx]
            min_v = float(np.min(train_slice))
            max_v = float(np.max(train_slice))
            if max_v == min_v:
                max_v = min_v + 1.0

            scaled = (cpu_values - min_v) / (max_v - min_v)
            self.data_per_vm.append(scaled)

        self.num_vms = len(self.data_per_vm)
        min_len = min(len(arr) for arr in self.data_per_vm)
        self.length = min_len - seq_len - (pred_len - 1)
        if self.length <= 0:
            raise ValueError(f"Not enough data. Min series length: {min_len}")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        seq = [
            torch.tensor(vm[idx : idx + self.seq_len]).unsqueeze(-1)
            for vm in self.data_per_vm
        ]
        seq_tensor = torch.stack(seq, dim=1)  # [seq_len, num_vms, 1]

        target = []
        for t in range(self.pred_len):
            target_t = torch.tensor(
                [vm[idx + self.seq_len + t] for vm in self.data_per_vm]
            ).unsqueeze(-1)
            target.append(target_t)
        target_tensor = torch.stack(target, dim=0)  # [pred_len, num_vms, 1]

        return seq_tensor, target_tensor


# -------------------------------------------------------------------------
# SHARED TRAINING UTILITIES
# -------------------------------------------------------------------------

class EarlyStopping:
    def __init__(self, patience=5, verbose=True):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"  EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True


def flatten_batch(sequences, targets):
    """Reshape [B, Seq, V, 1] -> [B*V, Seq, 1] to treat VMs as independent samples."""
    b, s, v, c = sequences.shape
    sequences = sequences.permute(0, 2, 1, 3).reshape(b * v, s, c)
    b_t, p, v_t, c_t = targets.shape
    targets = targets.permute(0, 2, 1, 3).reshape(b_t * v_t, p, c_t)
    return sequences, targets


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for sequences, targets in dataloader:
        sequences, targets = flatten_batch(sequences, targets)
        sequences = sequences.to(device)
        targets = targets.to(device).squeeze(-1)
        outputs = model(sequences)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for sequences, targets in dataloader:
            sequences, targets = flatten_batch(sequences, targets)
            sequences = sequences.to(device)
            targets = targets.to(device).squeeze(-1)
            outputs = model(sequences)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate_metrics(model, dataloader, device):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for sequences, targets in dataloader:
            sequences, targets = flatten_batch(sequences, targets)
            sequences = sequences.to(device)
            targets = targets.to(device).squeeze(-1)
            outputs = model(sequences)
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    preds = np.concatenate(all_preds).flatten()
    targs = np.concatenate(all_targets).flatten()

    mse = mean_squared_error(targs, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(targs, preds)
    r2 = r2_score(targs, preds)
    return mse, rmse, mae, r2


def get_dataloaders(data_dir, input_len, output_len, batch_size=32):
    first_csv = glob.glob(os.path.join(data_dir, "*.csv"))[0]
    total_time_steps = len(pd.read_csv(first_csv))
    train_end_idx = int(total_time_steps * 0.7)

    dataset = VMCPUDataset(
        csv_dir=data_dir,
        seq_len=input_len,
        pred_len=output_len,
        train_end_idx=train_end_idx,
    )

    n = len(dataset)
    train_size = int(n * 0.7)
    val_size = int(n * 0.15)
    train_idx = list(range(train_size))
    val_idx = list(range(train_size, train_size + val_size))
    test_idx = list(range(train_size + val_size, n))

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def run_training_cycle(model, train_loader, val_loader, device, num_epochs, lr, save_path):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    early_stopping = EarlyStopping(patience=5, verbose=True)
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        print(f"    Epoch {epoch+1}/{num_epochs} - Train: {train_loss:.6f} - Val: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)

        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("    Early stopping triggered")
            break

    model.load_state_dict(torch.load(save_path))
    return model


# -------------------------------------------------------------------------
# MODEL FACTORY
# -------------------------------------------------------------------------

# Per-model default learning rates (can be overridden via CLI)
MODEL_LR = {
    'bilstm':   {'lr': 0.001,   'ft_lr': 0.0001},
    'rnn':      {'lr': 0.001,   'ft_lr': 0.0001},
    'lstm':     {'lr': 0.001,   'ft_lr': 0.0001},
    'informer': {'lr': 0.0005,  'ft_lr': 0.00005},
    'tft':      {'lr': 0.001,   'ft_lr': 0.0001},
}


def build_model(model_name, output_len, input_len=None):
    if model_name == 'bilstm':
        return BiLSTMModel(input_size=1, output_size=output_len)
    elif model_name == 'rnn':
        return RNNModel(input_size=1, output_size=output_len)
    elif model_name == 'lstm':
        return LSTMModel(input_size=1, output_size=output_len)
    elif model_name == 'informer':
        return InformerModel(
            input_size=1, d_model=64, n_heads=4, n_layers=2,
            output_size=output_len, max_seq_len=input_len + 10
        )
    elif model_name == 'tft':
        return TemporalFusionTransformer(input_size=1, output_size=output_len)
    else:
        raise ValueError(f"Unknown model: {model_name}")


# -------------------------------------------------------------------------
# EXPERIMENT RUNNER
# -------------------------------------------------------------------------

CONFIGS = [
    (6, 1), (6, 2), (6, 3),
    (12, 1), (12, 2), (12, 3),
    (24, 1), (24, 2), (24, 3),
]


def run_experiment(model_name, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    lr = args.lr if args.lr is not None else MODEL_LR[model_name]['lr']
    ft_lr = args.ft_lr if args.ft_lr is not None else MODEL_LR[model_name]['ft_lr']

    f_azure = open(f"{model_name}_azure_results.txt", "w")
    f_alibaba = open(f"{model_name}_alibaba_results.txt", "w")
    f_azure_post = open(f"{model_name}_azure_post_ft_results.txt", "w")

    header = f"{'Input':<10} {'Output':<10} {'MSE':<10} {'RMSE':<10} {'MAE':<10} {'R2':<10}\n"
    for f in [f_azure, f_alibaba, f_azure_post]:
        f.write(header)
        f.flush()

    for in_len, out_len in CONFIGS:
        print(f"\n=== [{model_name.upper()}] Config: Input {in_len}, Output {out_len} ===")

        # Phase 1: Train on Azure, evaluate on Azure test set
        print("--- Phase 1: Training on Azure ---")
        train_loader, val_loader, test_loader = get_dataloaders(args.azure_dir, in_len, out_len)

        model = build_model(model_name, out_len, in_len).to(device)
        model_save_path = f"{model_name}_azure_{in_len}_{out_len}.pth"
        model = run_training_cycle(model, train_loader, val_loader, device, args.num_epochs, lr, model_save_path)

        mse, rmse, mae, r2 = evaluate_metrics(model, test_loader, device)
        f_azure.write(f"{in_len:<10} {out_len:<10} {mse:.6f}   {rmse:.6f}   {mae:.6f}   {r2:.6f}\n")
        f_azure.flush()
        print(f"Azure Results: MSE={mse:.6f}, R2={r2:.6f}")

        # Phase 2: Fine-tune on Alibaba, evaluate on Alibaba test set
        print("--- Phase 2: Fine-tuning on Alibaba ---")
        model = build_model(model_name, out_len, in_len).to(device)
        model.load_state_dict(torch.load(model_save_path))

        train_loader_ali, val_loader_ali, test_loader_ali = get_dataloaders(args.alibaba_dir, in_len, out_len)
        ft_save_path = f"{model_name}_alibaba_{in_len}_{out_len}.pth"
        model = run_training_cycle(model, train_loader_ali, val_loader_ali, device, args.num_epochs, ft_lr, ft_save_path)

        mse, rmse, mae, r2 = evaluate_metrics(model, test_loader_ali, device)
        f_alibaba.write(f"{in_len:<10} {out_len:<10} {mse:.6f}   {rmse:.6f}   {mae:.6f}   {r2:.6f}\n")
        f_alibaba.flush()
        print(f"Alibaba Results: MSE={mse:.6f}, R2={r2:.6f}")

        # Phase 3: Evaluate on Azure after fine-tuning (Catastrophic Forgetting check)
        print("--- Phase 3: Evaluating on Azure (Post-FT) ---")
        mse, rmse, mae, r2 = evaluate_metrics(model, test_loader, device)
        f_azure_post.write(f"{in_len:<10} {out_len:<10} {mse:.6f}   {rmse:.6f}   {mae:.6f}   {r2:.6f}\n")
        f_azure_post.flush()
        print(f"Azure Post-FT Results: MSE={mse:.6f}, R2={r2:.6f}")

    f_azure.close()
    f_alibaba.close()
    f_azure_post.close()
    print(f"\n[{model_name.upper()}] All configurations completed.")


# -------------------------------------------------------------------------
# ENTRY POINT
# -------------------------------------------------------------------------

ALL_MODELS = ['bilstm', 'rnn', 'lstm', 'informer', 'tft']


def main():
    parser = argparse.ArgumentParser(
        description="CPU Forecasting experiment runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --model all
  python main.py --model bilstm rnn
  python main.py --model informer --num_epochs 30 --azure_dir data/azure
        """
    )
    parser.add_argument(
        '--model', nargs='+', default=['all'],
        choices=['all'] + ALL_MODELS,
        help='Model(s) to run. Use "all" to run every model.'
    )
    parser.add_argument('--azure_dir', type=str, default='dataset/azure',
                        help='Path to Azure dataset directory')
    parser.add_argument('--alibaba_dir', type=str, default='dataset/alibaba',
                        help='Path to Alibaba dataset directory')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Maximum training epochs')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate override (default: per-model value)')
    parser.add_argument('--ft_lr', type=float, default=None,
                        help='Fine-tuning learning rate override (default: per-model value)')
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    models_to_run = ALL_MODELS if 'all' in args.model else args.model

    for model_name in models_to_run:
        print(f"\n{'='*60}")
        print(f"  STARTING EXPERIMENT: {model_name.upper()}")
        print(f"{'='*60}")
        run_experiment(model_name, args)

    print("\nAll experiments completed.")


if __name__ == "__main__":
    main()
