# CPU Forecasting — Journal Experiments

Multi-model CPU usage forecasting framework. Trains and evaluates deep learning models on Azure and Alibaba VM datasets across a 3-phase transfer learning protocol.

## Models

| Model | File |
|-------|------|
| BiLSTM | `bilstm/bilstm.py` |
| LSTM | `lstm/lstm.py` |
| RNN | `rnn/rnn.py` |
| Temporal Fusion Transformer (TFT) | `tft/tft.py` |
| Informer | `informer/informer.py` |
| ARIMA (baseline) | `arima/arima.py` |

## Experiment Protocol

For each model and each input/output configuration `(6-1, 6-2, 6-3, 12-1, 12-2, 12-3, 24-1, 24-2, 24-3)`:

1. **Phase 1** — Train on Azure → Evaluate on Azure test set
2. **Phase 2** — Fine-tune on Alibaba → Evaluate on Alibaba test set
3. **Phase 3** — Evaluate on Azure again (catastrophic forgetting check)

Results are saved to `<model>_azure_results.txt`, `<model>_alibaba_results.txt`, and `<model>_azure_post_ft_results.txt`.

## Dataset

```
dataset/
├── azure/       # 100 Azure VM CPU time series (.csv)
└── alibaba/     # 100 Alibaba VM CPU time series (.csv)
```

Each CSV must contain an `avg cpu` column (or use the first column as fallback).

## Usage

```bash
# Run all models
python main.py --model all

# Run specific models
python main.py --model bilstm rnn lstm

# Custom dataset paths and epochs
python main.py --model informer --azure_dir dataset/azure --alibaba_dir dataset/alibaba --num_epochs 30

# Override learning rates
python main.py --model tft --lr 0.0005 --ft_lr 0.00005
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `all` | Model(s) to run: `all`, `bilstm`, `rnn`, `lstm`, `informer`, `tft` |
| `--azure_dir` | `dataset/azure` | Path to Azure CSV directory |
| `--alibaba_dir` | `dataset/alibaba` | Path to Alibaba CSV directory |
| `--num_epochs` | `50` | Max training epochs (early stopping applies) |
| `--lr` | per-model | Learning rate override for Phase 1 training |
| `--ft_lr` | per-model | Learning rate override for Phase 2 fine-tuning |

### ARIMA baseline

```bash
python arima/arima.py
```

## Requirements

```
torch
numpy
pandas
scikit-learn
```

## Project Structure

```
.
├── main.py              # Experiment runner
├── dataset/
│   ├── azure/
│   └── alibaba/
├── bilstm/bilstm.py
├── lstm/lstm.py
├── rnn/rnn.py
├── tft/tft.py
├── informer/informer.py
└── arima/arima.py
```
