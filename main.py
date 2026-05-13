import os
import random
import time
import numpy as np
import torch
import pandas as pd

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

from data_processing import (
    load_and_preprocess, split_data, create_sequences, create_dataloaders,
)
from model import (
    ForecastModel, CNNLSTMAttention, SimpleMLP, SimpleLSTM, LEARModel,
)
from train import (
    train_dl_model, predict_dl, compute_metrics,
)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch, 'mps') and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def main():
    set_seed(42)
    device = get_device()
    print(f"Using device: {device}")

    data_path = 'Full_Dataset.csv'
    df, feature_cols = load_and_preprocess(data_path)
    n_features = len([c for c in feature_cols if c not in ('Price_BE', 'Price_CH')])
    print(f"Dataset: {len(df)} rows, {n_features} features")

    history_len = 24
    forecast_len = 72
    batch_size = 32
    split_ratio = (7, 2, 1)

    os.makedirs('results', exist_ok=True)

    data = split_data(df, feature_cols, split_ratio=split_ratio, normalize=False)
    X_train_seq, y_train_seq = create_sequences(
        data['X_train'], data['y_train'], history_len, forecast_len)
    X_val_seq, y_val_seq = create_sequences(
        data['X_val'], data['y_val'], history_len, forecast_len)
    X_test_seq, y_test_seq = create_sequences(
        data['X_test'], data['y_test'], history_len, forecast_len)

    print(f"Train: {X_train_seq.shape[0]}, Val: {X_val_seq.shape[0]}, Test: {X_test_seq.shape[0]}")

    train_loader = create_dataloaders(X_train_seq, y_train_seq, batch_size, shuffle=True)
    val_loader = create_dataloaders(X_val_seq, y_val_seq, batch_size, shuffle=False)
    test_loader = create_dataloaders(X_test_seq, y_test_seq, batch_size, shuffle=False)

    models_to_run = [
        # ('LEAR', None),
        # ('SimpleMLP', None),
        # ('SimpleLSTM', None),
        # ('CNN-BiLSTM-Attn', None),
        ('ForecastModel', 'predictions.csv'),
    ]
    results = []

    for model_name, csv_out in models_to_run:
        set_seed(42)
        print(f"\n--- {model_name} ---")
        t0 = time.time()

        if model_name == 'LEAR':
            model = LEARModel()
            model.fit(X_train_seq, y_train_seq)
            preds = model.predict(X_test_seq)
            targets = y_test_seq
        elif model_name == 'SimpleMLP':
            input_dim = history_len * n_features
            model = SimpleMLP(input_size=input_dim, hidden_size=128,
                              forecast_len=forecast_len).to(device)
            model, _, _ = train_dl_model(
                model, train_loader, val_loader, device,
                verbose=False)
            preds, targets = predict_dl(model, test_loader, device)
        elif model_name == 'SimpleLSTM':
            model = SimpleLSTM(input_size=n_features, hidden_size=64,
                               num_layers=2, forecast_len=forecast_len,
                               dropout=0.3).to(device)
            model, _, _ = train_dl_model(
                model, train_loader, val_loader, device,
                verbose=False)
            preds, targets = predict_dl(model, test_loader, device)
        elif model_name == 'CNN-BiLSTM-Attn':
            model = CNNLSTMAttention(input_size=n_features, cnn_channels=32,
                                     lstm_hidden=64, lstm_layers=2,
                                     forecast_len=forecast_len,
                                     dropout=0.3).to(device)
            model, _, _ = train_dl_model(
                model, train_loader, val_loader, device,
                verbose=False)
            preds, targets = predict_dl(model, test_loader, device)
        elif model_name == 'ForecastModel':
            model = ForecastModel(input_channels=n_features,
                                     history_len=history_len,
                                     forecast_len=forecast_len,
                                     window_sizes=[6, 8, 12]).to(device)
            model_path = os.path.join('pretrain_results', 'forecast_model.pth')
            model, _, _ = train_dl_model(
                model, train_loader, val_loader, device,
                verbose=False,
                model_path=model_path)
            preds, targets = predict_dl(model, test_loader, device)

        elapsed = time.time() - t0
        metrics = compute_metrics(targets, preds)
        metrics['Model'] = model_name
        metrics['Time_s'] = elapsed
        results.append(metrics)

        print(f"  MSE={metrics['MSE']:.4f}  MAE={metrics['MAE']:.4f}  "
              f"RMSE={metrics['RMSE']:.4f}  R2={metrics['R2']:.4f}  "
              f"wMAPE={metrics['wMAPE']:.4f}  sMAPE={metrics['sMAPE']:.4f}  "
              f"Time={elapsed:.1f}s")

        if csv_out is not None:
            final_true = y_test_seq[-1]
            final_pred = preds[-1]
            np.savetxt(os.path.join('pretrain_results', csv_out),
                       np.column_stack([final_true, final_pred]),
                       delimiter=',', fmt='%.4f',
                       header='true,pred', comments='')

    df_results = pd.DataFrame(results)[
        ['Model', 'MSE', 'MAE', 'RMSE', 'wMAPE', 'sMAPE', 'R2', 'Time_s']
    ]
    csv_path = os.path.join('pretrain_results', 'metrics.csv')
    df_results.to_csv(csv_path, index=False)

    print("\n" + "=" * 60)
    print("RESULTS (7:2:1 split)")
    print("=" * 60)
    print(df_results.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print(f"\nMetrics saved to {csv_path}")


if __name__ == '__main__':
    main()
