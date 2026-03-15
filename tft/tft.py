"""Temporal Fusion Transformer (TFT) model definition for CPU forecasting."""

import torch
import torch.nn as nn


class GatedResidualNetwork(nn.Module):
    """Gated Residual Network (GRN) — core building block of TFT."""

    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1, context_dim=None):
        super(GatedResidualNetwork, self).__init__()
        self.context_dim = context_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        if context_dim is not None:
            self.context_projection = nn.Linear(context_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(hidden_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.skip_layer = nn.Linear(input_dim, output_dim) if input_dim != output_dim else None

    def forward(self, x, context=None):
        h = self.fc1(x)
        if context is not None and self.context_dim is not None:
            h = h + self.context_projection(context)
        h = nn.functional.elu(h)
        h = nn.functional.elu(self.fc2(h))
        h = self.dropout(h)
        gate = torch.sigmoid(self.gate(h))
        h = gate * self.fc3(h)
        skip = self.skip_layer(x) if self.skip_layer is not None else x
        return self.layer_norm(skip + h)


class TemporalFusionTransformer(nn.Module):
    """Temporal Fusion Transformer for time series forecasting."""

    def __init__(self, input_size=1, hidden_size=128, num_heads=4, num_layers=2,
                 dropout=0.2, output_size=1):
        super(TemporalFusionTransformer, self).__init__()

        self.variable_selection = GatedResidualNetwork(
            input_dim=input_size, hidden_dim=hidden_size, output_dim=hidden_size, dropout=dropout
        )
        self.lstm_encoder = nn.LSTM(
            input_size=hidden_size, hidden_size=hidden_size,
            num_layers=num_layers, dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.post_attention_grn = GatedResidualNetwork(
            input_dim=hidden_size, hidden_dim=hidden_size, output_dim=hidden_size, dropout=dropout
        )
        self.output_grn = GatedResidualNetwork(
            input_dim=hidden_size, hidden_dim=hidden_size, output_dim=hidden_size, dropout=dropout
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: [batch, seq_len, input_size]
        x_selected = self.variable_selection(x)
        lstm_out, _ = self.lstm_encoder(x_selected)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.post_attention_grn(attn_out)
        out = self.layer_norm(lstm_out + attn_out)
        out = self.output_grn(out[:, -1, :])
        return self.output_layer(out)
