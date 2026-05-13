import os
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_processing import load_and_preprocess, split_data, create_sequences, create_dataloaders
from model import ForecastModel
from train import predict_dl

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from main import set_seed

def plot_features(data_path='Full_Dataset.csv', save_dir='pretrain_results'):
    df, feature_cols = load_and_preprocess(data_path)  # 假设已有加载函数

    n = len(df)
    t = np.arange(n)

    # 只保留两行子图
    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True,
                             gridspec_kw={'height_ratios': [1, 1]})

    # 第一图：Price_BE
    ax = axes[0]
    ax.plot(t, df['Price_BE'], linewidth=0.8, color='#1f77b4', label='Price_BE')  # 柔和蓝色
    ax.set_ylabel('Price (EUR/MWh)', fontsize=10)
    ax.set_title('Day-Ahead Electricity Belpex Price (EUR/MWh)', fontsize=11, fontweight='medium')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(loc='upper right', fontsize=8)

    # 第二图：Gen_BE 和 Load_BE
    ax = axes[1]
    ax.plot(t, df['Gen_BE'], linewidth=0.8, color='#2ca02c', label='Gen_BE')  # 青绿色
    ax.plot(t, df['Load_BE'], linewidth=0.8, color='#ff7f0e', label='Load_BE')  # 橙色
    ax.set_ylabel('Power (MW)', fontsize=10)
    ax.set_xlabel('Time index (hours)', fontsize=10)
    ax.set_title('Belgium Renewable Generation and Load (MW)', fontsize=11, fontweight='medium')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(loc='upper right', fontsize=8)

    fig.tight_layout(pad=1.5)
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, 'features.pdf')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Features plot saved to {path}")


def plot_forecast(save_dir='pretrain_results'):
    csv_path = os.path.join(save_dir, 'predictions.csv')
    df = pd.read_csv(csv_path)
    true = df['true'].values
    pred = df['pred'].values
    horizon = np.arange(len(true))

    # 增大图形尺寸：宽度 12 英寸，高度 6 英寸
    fig, ax = plt.subplots(figsize=(6, 4))

    # 真实值：蓝色实线 + 圆点
    ax.plot(horizon, true, 'o-', markersize=5, linewidth=1.5, color='blue', label='True')
    # 预测值：红色虚线 + 方块
    ax.plot(horizon, pred, 's--', markersize=5, linewidth=1.5, color='red', label='Predicted')

    ax.set_xlabel('Forecast horizon (hours)', fontsize=12)
    ax.set_ylabel('Price (EUR/MWh)', fontsize=12)
    ax.set_title('72-Hour Forecast: True vs Predicted', fontsize=14)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, linestyle='--', alpha=0.3)

    fig.tight_layout()
    path = os.path.join(save_dir, 'forecast.pdf')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Forecast plot saved to {path}")


def plot_test_results(save_dir='pretrain_results', data_path='Full_Dataset.csv'):
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')

    history_len = 24
    forecast_len = 72

    df, feature_cols = load_and_preprocess(data_path)
    n_features = len([c for c in feature_cols if c not in ('Price_BE', 'Price_CH')])
    set_seed(42)
    data = split_data(df, feature_cols, split_ratio=(7, 2, 1), normalize=False)
    X_test_seq, y_test_seq = create_sequences(
        data['X_test'], data['y_test'], history_len, forecast_len)
    test_loader = create_dataloaders(X_test_seq, y_test_seq, batch_size=32, shuffle=False)
    set_seed(42)
    model = ForecastModel(input_channels=n_features, history_len=history_len,
                          forecast_len=forecast_len, window_sizes=[6, 8, 12]).to(device)
    ckpt = torch.load(os.path.join(save_dir, 'forecast_model.pth'), map_location=device, weights_only=True)
    model.load_state_dict(ckpt)
    model.eval()

    preds, targets = predict_dl(model, test_loader, device)

    true = targets.flatten()
    pred = preds.flatten()

    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(true, pred, s=2, alpha=0.4, color='tab:blue', edgecolors='none')
    lim_min = min(true.min(), pred.min())
    lim_max = max(true.max(), pred.max())
    ax.plot([lim_min, lim_max], [lim_min, lim_max], '--', color='black', linewidth=0.8)
    ax.set_xlabel('True Price (EUR/MWh)')
    ax.set_ylabel('Predicted Price (EUR/MWh)')
    ax.set_title('ForecastModel: True vs Predicted on Test Set')
    ax.grid(True, alpha=0.3)
    ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes,
            fontsize=11, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    fig.tight_layout()
    path = os.path.join(save_dir, 'test_results.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Test results plot saved to {path}")


if __name__ == '__main__':
    plot_features()
    plot_forecast()
    plot_test_results()
