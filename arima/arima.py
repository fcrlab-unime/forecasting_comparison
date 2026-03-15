"""
ARIMA Baseline Experiment Script

Performs:
1. Eval on Azure   (fit ARIMA per sample) -> Save results
2. Eval on Alibaba (fit ARIMA per sample) -> Save results
3. Eval on Azure (Post-FT)               -> Identical to Phase 1
   (ARIMA has no global state, so fine-tuning has no effect)

Configurations: 6-1, 6-2, 6-3, 12-1, 12-2, 12-3, 24-1, 24-2, 24-3

Usage:
    python arima/arima.py
    python arima/arima.py --azure_dir ../datasets/azure --alibaba_dir ../datasets/alibaba
"""

import argparse
import glob
import os
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from torch.utils.data import DataLoader, Dataset, Subset

warnings.simplefilter('ignore', ConvergenceWarning)
warnings.simplefilter('ignore', UserWarning)

# -------------------------------------------------------------------------
# DATASET
# -------------------------------------------------------------------------

class VMCPUDataset(Dataset):
    """
    Loads per-VM CPU time series with min-max scaling fit on the training split.
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
# ARIMA EVALUATION
# -------------------------------------------------------------------------

def fit_predict_arima(history, pred_len, order=(2, 1, 0)):
    """Fit ARIMA on history and return a forecast of pred_len steps."""
    try:
        model_fit = ARIMA(history, order=order).fit()
        return model_fit.forecast(steps=pred_len)
    except Exception:
        return np.full(pred_len, history[-1])


def evaluate_arima_dataset(dataloader, pred_len):
    all_preds = []
    all_targets = []

    total = len(dataloader)
    for i, (sequences, targets) in enumerate(dataloader):
        if i % 10 == 0:
            print(f"    Processing batch {i}/{total}...", end='\r')

        # sequences: [B, Seq, V, 1] -> flat_history: [B*V, Seq]
        seq_np = sequences.numpy()
        tgt_np = targets.numpy()
        B, S, V, _ = seq_np.shape

        flat_history = seq_np.transpose(0, 2, 1, 3).reshape(-1, S)
        flat_target = tgt_np.transpose(0, 2, 1, 3).reshape(-1, pred_len)

        batch_preds = [fit_predict_arima(h, pred_len) for h in flat_history]
        all_preds.append(np.array(batch_preds))
        all_targets.append(flat_target)

    preds = np.concatenate(all_preds).flatten()
    targs = np.concatenate(all_targets).flatten()

    mse = mean_squared_error(targs, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(targs, preds)
    r2 = r2_score(targs, preds)
    return mse, rmse, mae, r2


# -------------------------------------------------------------------------
# DATA LOADING
# -------------------------------------------------------------------------

def get_test_loader(data_dir, input_len, output_len, batch_size=32):
    first_csv = glob.glob(os.path.join(data_dir, "*.csv"))[0]
    total_time_steps = len(pd.read_csv(first_csv))
    train_end_idx = int(total_time_steps * 0.7)

    dataset = VMCPUDataset(
        csv_dir=data_dir, seq_len=input_len, pred_len=output_len,
        train_end_idx=train_end_idx,
    )

    n = len(dataset)
    train_size = int(n * 0.7)
    val_size = int(n * 0.15)
    test_indices = list(range(train_size + val_size, n))

    return DataLoader(Subset(dataset, test_indices), batch_size=batch_size, shuffle=False)


# -------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------

CONFIGS = [
    (6, 1), (6, 2), (6, 3),
    (12, 1), (12, 2), (12, 3),
    (24, 1), (24, 2), (24, 3),
]


def main():
    parser = argparse.ArgumentParser(description="ARIMA Baseline Experiment")
    parser.add_argument('--azure_dir', type=str, default='../datasets/azure')
    parser.add_argument('--alibaba_dir', type=str, default='../datasets/alibaba')
    args = parser.parse_args()

    print("Running ARIMA Baseline Experiment")
    print("Note: ARIMA is a local statistical model with no global shared state.")
    print("Phase 3 (Post-FT) results are identical to Phase 1 by design.\n")

    f_azure = open("arima_azure_results.txt", "w")
    f_alibaba = open("arima_alibaba_results.txt", "w")
    f_azure_post = open("arima_azure_post_ft_results.txt", "w")

    header = f"{'Input':<10} {'Output':<10} {'MSE':<10} {'RMSE':<10} {'MAE':<10} {'R2':<10}\n"
    for f in [f_azure, f_alibaba, f_azure_post]:
        f.write(header)
        f.flush()

    for in_len, out_len in CONFIGS:
        print(f"\n=== Configuration: Input {in_len}, Output {out_len} ===")

        print("--- Phase 1: Evaluating on Azure ---")
        loader_az = get_test_loader(args.azure_dir, in_len, out_len)
        mse_az, rmse_az, mae_az, r2_az = evaluate_arima_dataset(loader_az, out_len)
        f_azure.write(f"{in_len:<10} {out_len:<10} {mse_az:.6f}   {rmse_az:.6f}   {mae_az:.6f}   {r2_az:.6f}\n")
        f_azure.flush()
        print(f"\nAzure Results: MSE={mse_az:.6f}, R2={r2_az:.6f}")

        print("--- Phase 2: Evaluating on Alibaba ---")
        loader_ali = get_test_loader(args.alibaba_dir, in_len, out_len)
        mse_ali, rmse_ali, mae_ali, r2_ali = evaluate_arima_dataset(loader_ali, out_len)
        f_alibaba.write(f"{in_len:<10} {out_len:<10} {mse_ali:.6f}   {rmse_ali:.6f}   {mae_ali:.6f}   {r2_ali:.6f}\n")
        f_alibaba.flush()
        print(f"\nAlibaba Results: MSE={mse_ali:.6f}, R2={r2_ali:.6f}")

        print("--- Phase 3: Azure (Post-FT) — same as Phase 1 ---")
        f_azure_post.write(f"{in_len:<10} {out_len:<10} {mse_az:.6f}   {rmse_az:.6f}   {mae_az:.6f}   {r2_az:.6f}\n")
        f_azure_post.flush()

    f_azure.close()
    f_alibaba.close()
    f_azure_post.close()
    print("\nAll experiments completed.")


if __name__ == "__main__":
    main()
