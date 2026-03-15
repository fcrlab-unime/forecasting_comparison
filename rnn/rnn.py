"""Vanilla RNN model definition for CPU forecasting."""

import torch.nn as nn


class RNNModel(nn.Module):
    """Standard vanilla RNN model for CPU forecasting."""

    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2, output_size=1):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False,
            nonlinearity='tanh',
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (batch_size, seq_length, input_size)
        rnn_out, _ = self.rnn(x)
        last_time_step = rnn_out[:, -1, :]
        return self.fc(last_time_step)
