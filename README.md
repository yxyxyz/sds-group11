# Electricity Price Prediction — SDS Group 11

Hour-ahead electricity price forecasting for the Belgian market, developed for the Smart Distribution Systems course (B-KUL-H00P3A) at KU Leuven (2026).

## Overview

The goal is to predict Belgian wholesale electricity prices (`Price_BE`) **72 hours ahead** using 24 hours of historical data, leveraging weather-driven generation forecasts (wind, solar) and load data.

## Dataset

| Column     | Description                          | Unit      |
|------------|--------------------------------------|-----------|
| `Date`     | Timestamp (hourly)                   | —         |
| `Price_BE` | Belgian electricity price (target)   | EUR/MWh   |
| `Wind_BE`  | Belgian wind generation forecast     | MW        |
| `Solar_BE` | Belgian solar generation forecast    | MW        |
| `Load_BE`  | Belgian electricity load             | MW        |
| `Load_FR`  | French electricity load              | MW        |
| `Gen_FR`   | French generation                    | MW        |
| `Price_CH` | Swiss electricity price              | EUR/MWh   |

Range: 2021-01-01 to 2026-02-25 (45,144 hourly samples).

## Project Structure

```
├── Full_Dataset.csv         # Input dataset (not tracked in git)
├── main.py                  # Entry point — runs all models and saves results
├── data_processing.py       # Data loading, cleaning, feature engineering
├── model.py                 # Neural network model architectures
├── train.py                 # Training loop, evaluation, and metrics
├── sds.ipynb                # Exploratory notebook (dual-path CNN-LSTM variant)
├── requirements_conda.yml   # Conda environment specification
├── results/                 # Model comparison metrics and predictions
├── Slides_SDS_ASSIGNMENT (2026).pdf  # Assignment reference
└── README.md
```

## Models

The pipeline compares 5 architectures:

| # | Model                   | Description                                    |
|---|-------------------------|------------------------------------------------|
| 1 | **LEAR**               | Lasso Estimated Auto-Regressive baseline       |
| 2 | **SimpleMLP**          | Multi-layer perceptron on flattened sequences  |
| 3 | **SimpleLSTM**         | 2-layer LSTM with Huber loss                   |
| 4 | **CNN-BiLSTM-Attn**    | 1D CNN + Bidirectional LSTM + attention        |
| 5 | **ForecastModel**   | Multi-scale STFT Imitation Network (SINCA)     |

Metric: MSE, MAE, RMSE, wMAPE, sMAPE, R².

## Setup

```bash
conda env create -f requirements_conda.yml
conda activate sds
```

## Usage

```bash
python main.py
```

Results are saved to `results/metrics.csv` (model comparison) and `results/predictions.csv` (forecasts vs actuals).

## Key Preprocessing Steps

- 5-sigma outlier clipping on all features
- Missing value imputation (linear interpolation → 24h shift-fill → forward/backward fill)
- Lag features: Price (24/48/72/168 h), Generation and Load (24/168 h)
- Sinusoidal day-of-week time features
- Chronological 70/20/10 train/val/test split
- Sliding windows: 24 h input → 72 h forecast horizon
