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


def plot_features(data_path='Full_Dataset.csv', save_dir='results'):
    df, feature_cols = load_and_preprocess(data_path)

    n = len(df)
    t = np.arange(n)

    rows = 5
    fig, axes = plt.subplots(rows, 1, figsize=(14, 3 * rows), sharex=True)

    # Row 1: Price_BE
    ax = axes[0]
    ax.plot(t, df['Price_BE'], linewidth=0.4, color='black')
    ax.set_ylabel('Price (EUR/MWh)')
    ax.set_title('Price_BE')

    # Row 2: Gen_BE and Load_BE
    ax = axes[1]
    ax.plot(t, df['Gen_BE'], linewidth=0.4, label='Gen_BE', color='darkgreen')
    ax.plot(t, df['Load_BE'], linewidth=0.4, label='Load_BE', color='darkorange')
    ax.set_ylabel('MW')
    ax.legend(fontsize=7, loc='upper right')
    ax.set_title('Gen_BE and Load_BE')

    # Row 3: Price lag features
    ax = axes[2]
    price_lag_cols = [c for c in df.columns if 'Price_BE_lag' in c]
    for col in price_lag_cols:
        ax.plot(t, df[col], linewidth=0.3, label=col, alpha=0.7)
    ax.set_ylabel('Price lag')
    ax.set_title('Price_BE Lags')
    ax.legend(fontsize=7, loc='upper right', ncol=2)

    # Row 4: Gen/Load lag features
    ax = axes[3]
    for var in ['Gen_BE', 'Load_BE']:
        lag_cols = [c for c in df.columns if f'{var}_lag' in c]
        for col in lag_cols:
            ax.plot(t, df[col], linewidth=0.3, label=col, alpha=0.7)
    ax.set_ylabel('MW lag')
    ax.set_title('Gen_BE and Load_BE Lags')
    ax.legend(fontsize=7, loc='upper right', ncol=2)

    # Row 5: day-of-week sinusoidal features
    ax = axes[4]
    zoom = 168 * 2
    ax.plot(t[:zoom], df['day_of_week_sin'].iloc[:zoom], linewidth=0.8, label='day_of_week_sin')
    ax.plot(t[:zoom], df['day_of_week_cos'].iloc[:zoom], linewidth=0.8, label='day_of_week_cos')
    ax.set_ylabel('Value')
    ax.set_title('Day-of-Week Features (first 2 weeks)')
    ax.set_xlabel('Time index (hours)')
    ax.legend(fontsize=8, loc='upper right')

    fig.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, 'features.pdf')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Features plot saved to {path}")


def plot_forecast(save_dir='results'):
    csv_path = os.path.join(save_dir, 'predictions.csv')
    df = pd.read_csv(csv_path)
    true = df['true'].values
    pred = df['pred'].values
    horizon = np.arange(len(true))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(horizon, true, 'o-', markersize=4, linewidth=1.2, label='True', color='black')
    ax.plot(horizon, pred, 's--', markersize=4, linewidth=1.2, label='Predicted', color='tab:red')
    ax.set_xlabel('Forecast horizon (hours)')
    ax.set_ylabel('Price (EUR/MWh)')
    ax.set_title('72-Hour Forecast: True vs Predicted')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(save_dir, 'forecast.pdf')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Forecast plot saved to {path}")


def plot_test_results(save_dir='results', data_path='Full_Dataset.csv'):
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')

    history_len = 24
    forecast_len = 72

    df, feature_cols = load_and_preprocess(data_path)
    n_features = len([c for c in feature_cols if c not in ('Price_BE', 'Price_CH')])

    data = split_data(df, feature_cols, split_ratio=(7, 2, 1), normalize=False)
    X_test_seq, y_test_seq = create_sequences(
        data['X_test'], data['y_test'], history_len, forecast_len)
    test_loader = create_dataloaders(X_test_seq, y_test_seq, batch_size=32, shuffle=False)

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

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(true, pred, s=2, alpha=0.4, color='tab:blue', edgecolors='none')
    lim_min = min(true.min(), pred.min())
    lim_max = max(true.max(), pred.max())
    ax.plot([lim_min, lim_max], [lim_min, lim_max], '--', color='black', linewidth=0.8)
    ax.set_xlabel('True Price (EUR/MWh)')
    ax.set_ylabel('Predicted Price (EUR/MWh)')
    ax.set_title('ForecastModel: True vs Predicted on Test Set')
    ax.grid(True, alpha=0.3)
    ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes,
            fontsize=11, va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    fig.tight_layout()
    path = os.path.join(save_dir, 'test_results.pdf')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Test results plot saved to {path}")


if __name__ == '__main__':
    plot_features()
    plot_forecast()
    plot_test_results()
