import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)


def train_dl_model(model, train_loader, val_loader, device,
                   lr=1e-3, weight_decay=1e-5, num_epochs=1000,
                   early_stop_patience=50, model_path=None, verbose=True):
    criterion = nn.HuberLoss(delta=5.0)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            if verbose:
                print(f"Epoch {epoch+1}: new best, val_loss={val_loss:.6f}")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                if verbose:
                    print(f"Early stop at epoch {epoch+1}")
                break

        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} | train: {train_loss:.6f} | val: {val_loss:.6f}")

    if model_path is not None:
        torch.save(best_state, model_path)
    model.load_state_dict(best_state)
    model.eval()
    return model, train_losses, val_losses


def predict_dl(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy()
            all_preds.append(pred)
            all_targets.append(yb.cpu().numpy())
    return np.concatenate(all_preds, axis=0), np.concatenate(all_targets, axis=0)


def compute_metrics(targets, preds):
    mae = mean_absolute_error(targets, preds)
    mse = mean_squared_error(targets, preds)
    rmse = np.sqrt(mse)
    wmape = np.sum(np.abs(preds - targets)) / (np.sum(np.abs(targets)) + 1e-8)
    epsilon = 1e-8
    smape = np.mean(2.0 * np.abs(preds - targets) / (np.abs(preds) + np.abs(targets) + epsilon))
    r2 = r2_score(targets, preds)

    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'wMAPE': wmape,
        'sMAPE': smape,
        'R2': r2,
    }
