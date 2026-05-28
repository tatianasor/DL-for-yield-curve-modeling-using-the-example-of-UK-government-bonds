import os
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.api import VAR
from statsmodels.tsa.ar_model import AutoReg

# -----------------------------
# 1. LOAD DATA
# -----------------------------
train = pd.read_csv("UK_yield_curve_daily_processed_train.csv")
val = pd.read_csv("UK_yield_curve_daily_processed_val.csv")
test = pd.read_csv("UK_yield_curve_daily_processed_test.csv")

maturities = train.columns[1:]

datasets = {
    "Train": train[maturities],
    "Val": val[maturities],
    "Test": test[maturities]
}

# -----------------------------
# 2. METRICS FUNCTION
# -----------------------------
def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2}

# -----------------------------
# 3. RANDOM WALK (t → t+1)
# -----------------------------
def random_walk_static(data):
    y = data.values
    y_true = y[1:]
    y_pred = y[:-1]
    return calculate_metrics(y_true, y_pred)

# -----------------------------
# 4. VAR(1) ON PCA FACTORS (t → t+1)
# -----------------------------
def var_forecast_static(data, n_factors=3):
    y = data.values

    # PCA только на X_t
    pca = PCA(n_components=n_factors)
    factors = pca.fit_transform(y[:-1])

    y_true = y[1:]

    var_model = VAR(factors)
    var_results = var_model.fit(maxlags=1)

    y_pred_factors = []
    for t in range(len(factors)-1):
        forecast = var_results.forecast(factors[t:t+1], steps=1)[0]
        y_pred_factors.append(forecast)

    y_pred_factors = np.array(y_pred_factors)

    # Обратное преобразование в yield space
    y_pred = y_pred_factors @ pca.components_ + pca.mean_
    y_true = y_true[:len(y_pred)]

    return calculate_metrics(y_true, y_pred)

# -----------------------------
# 5. Diebold-Li simplified (AR1 on PCA factors)
# -----------------------------
def diebold_li_forecast_static(data, n_factors=3):
    y = data.values

    pca = PCA(n_components=n_factors)
    factors = pca.fit_transform(y[:-1])

    y_true = y[1:]

    factors_pred = []

    for i in range(n_factors):
        ar_model = AutoReg(factors[:, i], lags=1, old_names=False)
        ar_res = ar_model.fit()
        pred = ar_res.predict(start=1, end=len(factors)-1)
        factors_pred.append(pred)

    factors_pred = np.stack(factors_pred, axis=1)

    # Обратное преобразование
    y_pred = factors_pred @ pca.components_ + pca.mean_
    y_true = y_true[:len(y_pred)]

    return calculate_metrics(y_true, y_pred)

# -----------------------------
# 6. RUN ALL MODELS
# -----------------------------
rw_results = {}
var_results = {}
dl_results = {}

for name, data in datasets.items():
    print(f"\nProcessing {name} set...")
    rw_results[name] = random_walk_static(data)
    var_results[name] = var_forecast_static(data)
    dl_results[name] = diebold_li_forecast_static(data)

# -----------------------------
# 7. CONVERT TO DATAFRAMES (FIXED)
# -----------------------------
rw_df = pd.DataFrame([
    {'Dataset': name, **metrics} for name, metrics in rw_results.items()
]).set_index('Dataset')

var_df = pd.DataFrame([
    {'Dataset': name, **metrics} for name, metrics in var_results.items()
]).set_index('Dataset')

dl_df = pd.DataFrame([
    {'Dataset': name, **metrics} for name, metrics in dl_results.items()
]).set_index('Dataset')

# -----------------------------
# 8. PRINT TABLES
# -----------------------------
print("\n=== STATIC FORECAST (Random Walk) ===")
print(rw_df)

print("\n=== STATIC FORECAST (VAR1_PCA) ===")
print(var_df)

print("\n=== STATIC FORECAST (DieboldLi_NS) ===")
print(dl_df)