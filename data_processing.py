import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler


def load_and_preprocess(data_path):
    df_all = pd.read_csv(data_path, header=0)
    df_all.rename(columns={df_all.columns[0]: 'time'}, inplace=True)
    df_all['time'] = pd.to_datetime(df_all['time'], dayfirst=True)
    df_all.set_index('time', inplace=True)

    df_all['Gen_BE'] = df_all['Solar_BE'] + df_all['Wind_BE']
    feature_cols = ['Price_BE', 'Gen_BE', 'Load_BE']
    df = df_all[feature_cols].copy()

    price_mean = df['Price_BE'].mean()
    price_std = df['Price_BE'].std()
    upper_limit = price_mean + 3 * price_std
    lower_limit = price_mean - 3 * price_std
    df.loc[df['Price_BE'] > upper_limit, 'Price_BE'] = upper_limit
    df.loc[df['Price_BE'] < lower_limit, 'Price_BE'] = lower_limit

    for col in feature_cols:
        df[col] = df[col].interpolate(method='linear', limit=3)
        df[col] = df[col].fillna(df[col].shift(24))
        df[col] = df[col].bfill().ffill()

    price_lags = [24, 48, 72]
    for lag in price_lags:
        df[f'Price_BE_lag_{lag}'] = df['Price_BE'].shift(lag)
        feature_cols.append(f'Price_BE_lag_{lag}')

    other_lags = [24, 72]
    for var in ['Gen_BE', 'Load_BE']:
        for lag in other_lags:
            df[f'{var}_lag_{lag}'] = df[var].shift(lag)
            feature_cols.append(f'{var}_lag_{lag}')

    # df['day_of_week_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    # df['day_of_week_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)

    # time_features = ['day_of_week_sin', 'day_of_week_cos']
    # feature_cols.extend(time_features)

    df = df.dropna()
    df = df.reset_index(drop=True)

    return df, feature_cols


def split_data(df, feature_cols, split_ratio=(7, 2, 1), normalize=False):
    n = len(df)
    s1 = int(split_ratio[0] / sum(split_ratio) * n)
    s2 = int((split_ratio[0] + split_ratio[1]) / sum(split_ratio) * n)

    train_df = df.iloc[:s1].copy()
    val_df = df.iloc[s1:s2].copy()
    test_df = df.iloc[s2:].copy()

    feature_cols_noprice = [c for c in feature_cols if c not in ('Price_BE', 'Price_CH')]

    X_train = train_df[feature_cols_noprice].values.astype(np.float32)
    y_train = train_df['Price_BE'].values.reshape(-1, 1).astype(np.float32)
    X_val = val_df[feature_cols_noprice].values.astype(np.float32)
    y_val = val_df['Price_BE'].values.reshape(-1, 1).astype(np.float32)
    X_test = test_df[feature_cols_noprice].values.astype(np.float32)
    y_test = test_df['Price_BE'].values.reshape(-1, 1).astype(np.float32)

    scaler_X, scaler_y = None, None
    if normalize:
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_train = scaler_X.fit_transform(X_train).astype(np.float32)
        y_train = scaler_y.fit_transform(y_train).astype(np.float32)
        X_val = scaler_X.transform(X_val).astype(np.float32)
        y_val = scaler_y.transform(y_val).astype(np.float32)
        X_test = scaler_X.transform(X_test).astype(np.float32)
        y_test = scaler_y.transform(y_test).astype(np.float32)

    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'train_df': train_df, 'val_df': val_df, 'test_df': test_df,
        'feature_cols': feature_cols_noprice,
        'scaler_X': scaler_X, 'scaler_y': scaler_y,
    }


def create_sequences(X, y, history_len=24, forecast_len=72):
    X_seq, y_seq = [], []
    total = len(X) - history_len - forecast_len + 1
    for i in range(total):
        X_seq.append(X[i:i + history_len, :])
        y_seq.append(y[i + history_len:i + history_len + forecast_len].flatten())
    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32)


def create_dataloaders(X_seq, y_seq, batch_size=32, shuffle=True):
    X_t = torch.tensor(X_seq, dtype=torch.float32)
    y_t = torch.tensor(y_seq, dtype=torch.float32)
    dataset = TensorDataset(X_t, y_t)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
