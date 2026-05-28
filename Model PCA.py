import joblib
import pandas as pd
import numpy as np
import os

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# -----------------------------
# 1. LOAD DATA
# -----------------------------
train = pd.read_csv("UK_yield_curve_daily_processed_train.csv")
val   = pd.read_csv("UK_yield_curve_daily_processed_val.csv")
test  = pd.read_csv("UK_yield_curve_daily_processed_test.csv")

maturities = train.columns[1:]

# -----------------------------
# ЕДИНЫЙ Y_TRUE ДЛЯ ТЕСТА
# Берём напрямую из исходного файла, сдвиг t→t+1
# -----------------------------
Y_true_universal = test[maturities].values[1:]  # строки 1..N

# -----------------------------
# METRICS
# -----------------------------
def calculate_metrics(y_true, y_pred):
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    return mse, rmse, mae, r2

# -----------------------------
# STATIC FORECAST FUNCTION
# Возвращает y_pred; y_true для train/val считаем внутри
# для test используем Y_true_universal
# -----------------------------
def static_forecast_pred(data, pca_model):
    """Возвращает только y_pred (прогноз на t+1)."""
    X = data[:-1]
    factors = pca_model.transform(X)
    lr = LinearRegression()
    lr.fit(factors, data[1:])
    y_pred = lr.predict(factors)
    return y_pred

def static_forecast_full(data, pca_model):
    """Для train/val: возвращает y_true и y_pred через PCA."""
    X        = data[:-1]
    y_target = data[1:]
    factors  = pca_model.transform(X)
    lr = LinearRegression()
    lr.fit(factors, y_target)
    y_pred = lr.predict(factors)
    return y_target, y_pred

# -----------------------------
# GRID SEARCH по числу компонент
# -----------------------------
print("\nGRID SEARCH PCA components")
best_r2 = -np.inf
best_n  = None

for n in [3, 5, 7, 10, 15]:
    pca_tmp = PCA(n_components=min(n, len(maturities)))
    pca_tmp.fit(train[maturities].values)

    y_true_val, y_pred_val = static_forecast_full(
        val[maturities].values, pca_tmp
    )
    r2 = r2_score(y_true_val, y_pred_val)
    print(f"n_components={n}  VAL R2={r2:.4f}")

    if r2 > best_r2:
        best_r2 = r2
        best_n  = n

print("\nBEST PCA COMPONENTS:", best_n)

# -----------------------------
# FINAL PCA
# -----------------------------
pca = PCA(n_components=best_n)
pca.fit(train[maturities].values)

os.makedirs("artifacts_pca", exist_ok=True)
joblib.dump(pca, "artifacts_pca/pca_model.pkl")

# -----------------------------
# СОХРАНЕНИЯ ДЛЯ ШАГА 14
# -----------------------------
pd.DataFrame(
    pca.components_, columns=maturities
).to_csv("artifacts_pca/pca_components.csv", index=False)

pd.DataFrame(
    pca.explained_variance_ratio_,
    columns=["explained_variance"]
).to_csv("artifacts_pca/pca_explained_variance.csv", index=False)

Z_train = pca.transform(train[maturities].values)
Z_val   = pca.transform(val[maturities].values)
Z_test  = pca.transform(test[maturities].values)

pd.DataFrame(Z_train).to_csv("artifacts_pca/pca_factors_train.csv", index=False)
pd.DataFrame(Z_val).to_csv("artifacts_pca/pca_factors_val.csv",   index=False)
pd.DataFrame(Z_test).to_csv("artifacts_pca/pca_factors_test.csv",  index=False)

# -----------------------------
# PREDICTIONS
# -----------------------------
Y_train_true, Y_train_pred = static_forecast_full(
    train[maturities].values, pca
)
Y_val_true, Y_val_pred = static_forecast_full(
    val[maturities].values, pca
)

# Для теста: y_pred из модели, y_true — исходные данные
Y_test_pred = static_forecast_pred(test[maturities].values, pca)
Y_test_true = Y_true_universal  # единый y_true

# Проверка размеров
assert len(Y_test_true) == len(Y_test_pred), \
    f"Размеры не совпадают: {len(Y_test_true)} vs {len(Y_test_pred)}"

# -----------------------------
# METRICS TABLE
# -----------------------------
results = pd.DataFrame(
    [
        calculate_metrics(Y_train_true, Y_train_pred),
        calculate_metrics(Y_val_true,   Y_val_pred),
        calculate_metrics(Y_test_true,  Y_test_pred)
    ],
    columns=["MSE", "RMSE", "MAE", "R2"],
    index=["Train", "Val", "Test"]
)

print("\n=== STATIC FORECAST (PCA → Linear Regression) ===")
print(results)

# -----------------------------
# SAVE 9 CSV FILES
# -----------------------------
os.makedirs("predictions", exist_ok=True)

def save(prefix, y_true, y_pred):
    residuals = y_true - y_pred
    pd.DataFrame(y_true).to_csv(
        f"predictions/{prefix}_y_true.csv", index=False)
    pd.DataFrame(y_pred).to_csv(
        f"predictions/{prefix}_y_pred.csv", index=False)
    pd.DataFrame(residuals).to_csv(
        f"predictions/{prefix}_residuals.csv", index=False)

save("PCA_LR_train", Y_train_true, Y_train_pred)
save("PCA_LR_val",   Y_val_true,   Y_val_pred)
save("PCA_LR_test",  Y_test_true,  Y_test_pred)

# -----------------------------
# SAVE METRICS TO EXCEL
# -----------------------------
results.to_excel("predictions/results_PCA_LR.xlsx")
print("Metrics saved: predictions/results_PCA_LR.xlsx")