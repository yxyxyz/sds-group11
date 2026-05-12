import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LassoCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler


# --- STFT Imitation Network (single scale) ---
class SpectralInceptionNet(nn.Module):
    def __init__(self, input_length=72, window_size=24):
        super().__init__()
        self.window_size = window_size
        self.stride = window_size // 2
        self.fft_bins = window_size // 2 + 1
        self.time_steps = math.ceil(input_length / self.stride) + 1
        self.padding = int(((self.time_steps - 1) * self.stride + window_size - input_length) / 2)

        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=2 * self.fft_bins,
            kernel_size=self.window_size,
            stride=self.stride,
            padding=self.padding,
            bias=False
        )

    def forward(self, x):
        B, L, C = x.shape
        x = x.permute(0, 2, 1)
        x = x.reshape(B * C, 1, L)
        x = self.conv(x)
        x = x.view(B, C, 2 * self.fft_bins, -1)

        real = x[:, :, 0::2, :]
        imag = x[:, :, 1::2, :]

        mag = torch.sqrt(real ** 2 + imag ** 2 + 1e-8)
        return mag


# --- Multi-Scale STFT with ConvTranspose2d upsampling (DETERMINISTIC) ---
class MultiScaleSINE(nn.Module):
    def __init__(self, input_length=72, window_sizes=[6, 8, 12]):
        super().__init__()
        self.window_sizes = window_sizes
        self.stft_layers = nn.ModuleList()

        # Compute per-scale spatial dims and target (largest) dims
        self.freq_dims = []
        self.time_dims = []
        for ws in window_sizes:
            stride = ws // 2
            fft_bins = ws // 2 + 1
            time_steps = math.ceil(input_length / stride) + 1
            self.freq_dims.append(fft_bins)
            self.time_dims.append(time_steps)

            conv_layer = nn.Conv1d(
                in_channels=1,
                out_channels=2 * fft_bins,
                kernel_size=ws,
                stride=stride,
                padding=int(((time_steps - 1) * stride + ws - input_length) / 2),
                bias=False
            )
            self.stft_layers.append(conv_layer)

    def forward(self, x):
        B, L, C = x.shape
        outputs = []

        x_reshaped = x.permute(0, 2, 1)
        x_reshaped = x_reshaped.reshape(B * C, 1, L)

        for conv_layer in self.stft_layers:
            out = conv_layer(x_reshaped)
            out = out.view(B, C, -1, out.shape[-1])
            real = out[:, :, 0::2, :]
            imag = out[:, :, 1::2, :]
            mag = torch.sqrt(real ** 2 + imag ** 2 + 1e-8)
            outputs.append(mag)

        return outputs


# --- Channel Attention ---
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(B, C))
        max_out = self.fc(self.max_pool(x).view(B, C))
        attention = torch.sigmoid(avg_out + max_out)
        return x * attention.unsqueeze(2).unsqueeze(3)


# --- Spatial Attention ---
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(out)) * x


# --- Residual Conv2d ---
class ResidualConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, dilation=1, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.shortcut = nn.Identity() if in_channels == out_channels and stride == 1 else \
            nn.Conv2d(in_channels, out_channels, 1, stride=stride)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out + self.shortcut(x)


# --- ForecastModel (Multi-scale STFT + ConvTranspose2d upsampling) ---
class ForecastModel(nn.Module):
    def __init__(self, input_channels=12, history_len=24, forecast_len=72,
                 window_sizes=[6, 8, 12]):
        super().__init__()
        self.window_sizes = window_sizes
        self.input_channels = input_channels
        self.history_len = history_len

        self.multi_scale_stft = MultiScaleSINE(
            input_length=history_len, window_sizes=window_sizes
        )

        self.target_H = max(ws // 2 + 1 for ws in window_sizes)
        self.target_W = max(math.ceil(history_len / (ws // 2)) + 1 for ws in window_sizes)

        total_channels = input_channels * len(window_sizes)

        self.cnn = nn.Sequential(
            ResidualConv2d(total_channels, 32, kernel_size=3, dilation=2, padding=2),
            # nn.Conv2d(total_channels, 32, kernel_size=3, dilation=2, padding=2),
            #nn.BatchNorm2d(32),
            #nn.ReLU(),
            ChannelAttention(32),
            # SpatialAttention(),
            nn.MaxPool2d((2, 1)),

            ResidualConv2d(32, 64, kernel_size=3, padding=1),
            #nn.Conv2d(32, 64, kernel_size=(3, 3), dilation=1, padding=(1, 1)),
            # nn.BatchNorm2d(64),
            # nn.ReLU(),
            ChannelAttention(64),
            # SpatialAttention(),
            nn.MaxPool2d((2, 2))
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.Linear(128, forecast_len)
        )

    def forward(self, x):
        # x: (B, T, C)
        specs = self.multi_scale_stft(x)

        upsampled = []
        for spec in specs:
            upsampled.append(F.interpolate(spec, size=(self.target_H, self.target_W),
                                           mode='nearest'))
        combined = torch.cat(upsampled, dim=1)  # (B, C*len(ws), H_target, W_target)

        features = self.cnn(combined)
        output = self.classifier(features)
        return output


# --- CNN-BiLSTM-Attention ---
class CNNLSTMAttention(nn.Module):
    def __init__(self, input_size, cnn_channels=32, cnn_kernel=3,
                 lstm_hidden=64, lstm_layers=2, forecast_len=24, dropout=0.3):
        super().__init__()
        self.cnn_conv = nn.Conv1d(in_channels=input_size, out_channels=cnn_channels,
                                  kernel_size=cnn_kernel, padding=cnn_kernel // 2)

        self.lstm = nn.LSTM(input_size=cnn_channels, hidden_size=lstm_hidden,
                            num_layers=lstm_layers, batch_first=True,
                            bidirectional=True, dropout=dropout)

        lstm_out_dim = lstm_hidden * 2
        self.attn = nn.Linear(lstm_out_dim, 1)
        self.fc = nn.Sequential(
            nn.Linear(lstm_out_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, forecast_len)
        )

    def forward(self, x):
        # x: (B, T, C)
        x = x.permute(0, 2, 1)
        cnn_out = F.relu(self.cnn_conv(x))
        cnn_out = cnn_out.permute(0, 2, 1)
        lstm_out, _ = self.lstm(cnn_out)
        attn_weights = torch.softmax(self.attn(lstm_out), dim=1)
        context = (lstm_out * attn_weights).sum(dim=1)
        out = self.fc(context)
        return out


# --- Simple MLP ---
class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size=128, forecast_len=24):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, forecast_len)
        )

    def forward(self, x):
        # x: (B, T, C)
        x = x.view(x.size(0), -1)
        return self.mlp(x)


# --- Simple LSTM ---
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, forecast_len=24, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, forecast_len)
        )

    def forward(self, x):
        # x: (B, T, C)
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out


# --- LEAR (Lasso-based) ---
class LEARModel:
    def __init__(self, alpha=None, cv=3, max_iter=5000):
        self.base_lasso = LassoCV(cv=cv, random_state=42, n_jobs=4, max_iter=max_iter)
        self.model = MultiOutputRegressor(self.base_lasso, n_jobs=4)
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()

    def fit(self, X_train, y_train):
        X_flat = X_train.reshape(X_train.shape[0], -1)
        X_scaled = self.scaler_X.fit_transform(X_flat)
        y_scaled = self.scaler_y.fit_transform(y_train)
        self.model.fit(X_scaled, y_scaled)
        return self

    def predict(self, X):
        X_flat = X.reshape(X.shape[0], -1)
        X_scaled = self.scaler_X.transform(X_flat)
        preds_scaled = self.model.predict(X_scaled)
        return self.scaler_y.inverse_transform(preds_scaled)
