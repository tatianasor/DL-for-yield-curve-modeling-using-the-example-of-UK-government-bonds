import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ===============================
# PATHS
# ===============================
PRED_DIR = "predictions"
TEST_PATH = "UK_yield_curve_daily_processed_test.csv"
OUT_DIR   = "artifacts_step12"
os.makedirs(OUT_DIR, exist_ok=True)

# ===============================
# 1. ЗАГРУЗКА ДАТ ИЗ ТЕСТОВОГО ФАЙЛА
# ===============================
test = pd.read_csv(TEST_PATH)
test["Date"] = pd.to_datetime(test.iloc[:, 0])
test_dates = test["Date"].values  # все даты теста

print(f"Test период: {test['Date'].min().date()} — {test['Date'].max().date()}")
print(f"Test строк: {len(test)}")

# ===============================
# 2. ЛУЧШИЕ ВЕРСИИ МОДЕЛЕЙ (по seeds)
# ===============================
# Формат: (prefix, seed, n_skip)
# n_skip = сколько строк теряется в начале теста
# PCA/AE_MLP/AE_LSTM/AE_dense: теряют 1 строку (t→t+1)
# PCA_GRU/AE_GRU: теряют LOOKBACK+1=6 строк
# Начальные даты:
# PCA, PCA_MLP, PCA_LSTM, AE_MLP, AE_LSTM: 2020-02-18 → n_skip=1
# AE_dense: 2020-02-21 → n_skip=4
# PCA_GRU, AE_GRU: 2020-02-23 → n_skip=6

models = {
    "PCA_LR":   {"prefix": "PCA_LR",   "seed": None, "n_skip": 1},
    "AE_dense": {"prefix": "AE_dense", "seed": 42,   "n_skip": 4},
    "PCA_MLP":  {"prefix": "PCA_MLP",  "seed": 42,   "n_skip": 1},
    "PCA_LSTM": {"prefix": "PCA_LSTM", "seed": 1,    "n_skip": 1},
    "PCA_GRU":  {"prefix": "PCA_GRU",  "seed": 1,    "n_skip": 6},
    "AE_MLP":   {"prefix": "AE_MLP",   "seed": 123,  "n_skip": 1},
    "AE_LSTM":  {"prefix": "AE_LSTM",  "seed": 123,  "n_skip": 1},
    "AE_GRU":   {"prefix": "AE_GRU",   "seed": 2025, "n_skip": 6},
}

# ===============================
# 3. ЗАГРУЗКА Y_PRED ДЛЯ КАЖДОЙ МОДЕЛИ
# ===============================
def load_pred(prefix, seed):
    if seed is None:
        fname = f"{PRED_DIR}/{prefix}_test_y_pred.csv"
    else:
        fname = f"{PRED_DIR}/{prefix}_test_y_pred_{seed}.csv"
    df = pd.read_csv(fname)
    return df.values  # (N, 50)

# Загружаем все предсказания и привязываем даты
model_data = {}

for name, cfg in models.items():
    y_pred = load_pred(cfg["prefix"], cfg["seed"])
    n_skip = cfg["n_skip"]

    # Даты для этой модели: test_dates[n_skip:]
    dates = test_dates[n_skip:n_skip + len(y_pred)]

    # Y_true: исходные данные теста с той же датой
    maturities = test.columns[1:]
    y_true = test[maturities].values[n_skip:n_skip + len(y_pred)]

    model_data[name] = {
        "dates":  pd.to_datetime(dates),
        "y_true": y_true,
        "y_pred": y_pred,
    }

    print(f"{name}: {len(y_pred)} строк, "
          f"с {pd.to_datetime(dates[0]).date()} "
          f"по {pd.to_datetime(dates[-1]).date()}")

# ===============================
# 4. СИНХРОНИЗАЦИЯ — общая начальная дата
# Самая поздняя начальная дата = 2020-02-23 (PCA_GRU, AE_GRU)
# ===============================
# Находим максимальную начальную дату среди всех моделей
start_dates = {name: model_data[name]["dates"].min()
               for name in model_data}
common_start = max(start_dates.values())

print(f"\nОбщая начальная дата: {common_start.date()}")

# Обрезаем все модели с common_start
for name in model_data:
    d = model_data[name]
    mask = d["dates"] >= common_start
    model_data[name]["dates"]  = d["dates"][mask]
    model_data[name]["y_true"] = d["y_true"][mask]
    model_data[name]["y_pred"] = d["y_pred"][mask]
    print(f"{name}: после синхронизации {mask.sum()} строк, "
          f"с {model_data[name]['dates'].min().date()}")

# ===============================
# 5. РЕЖИМЫ РЫНКА
# ===============================
regimes = {
    "COVID_2020_2021":         ("2020-02-23", "2021-12-31"),
    "MiniBudget_2022":         ("2022-01-01", "2022-12-31"),
    "Normalization_2023_2025": ("2023-01-01", "2025-12-31"),
    "New_2026":                ("2026-01-01", "2026-04-30"),
}

regime_labels = {
    "COVID_2020_2021":         "COVID 2020–2021",
    "MiniBudget_2022":         "Mini-Budget 2022",
    "Normalization_2023_2025": "Нормализация 2023–2025",
    "New_2026":                "2026 (новейший)",
}

# ===============================
# 6. МЕТРИКИ
# ===============================
def calc_metrics(y_true, y_pred):
    if len(y_true) == 0:
        return {"MSE": np.nan, "RMSE": np.nan,
                "MAE": np.nan, "R2": np.nan, "N": 0}
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2, "N": len(y_true)}

# ===============================
# 7. РАСЧЁТ МЕТРИК ПО РЕЖИМАМ
# ===============================
print("\n=== МЕТРИКИ ПО РЕЖИМАМ ===")

for regime_key, (start, end) in regimes.items():
    label  = regime_labels[regime_key]
    start_ = pd.Timestamp(start)
    end_   = pd.Timestamp(end)

    rows = []
    for name in models.keys():
        d    = model_data[name]
        mask = (d["dates"] >= start_) & (d["dates"] <= end_)

        y_true_r = d["y_true"][mask]
        y_pred_r = d["y_pred"][mask]

        m = calc_metrics(y_true_r, y_pred_r)
        rows.append({
            "Model": name,
            "N":     m["N"],
            "MSE":   round(m["MSE"],  6) if not np.isnan(m["MSE"])  else np.nan,
            "RMSE":  round(m["RMSE"], 6) if not np.isnan(m["RMSE"]) else np.nan,
            "MAE":   round(m["MAE"],  6) if not np.isnan(m["MAE"])  else np.nan,
            "R2":    round(m["R2"],   6) if not np.isnan(m["R2"])   else np.nan,
        })

    # Сортируем по RMSE (возрастание = лучший первый)
    df_regime = pd.DataFrame(rows).sort_values("RMSE")
    df_regime = df_regime.reset_index(drop=True)
    df_regime.index += 1  # ранг с 1

    print(f"\n--- {label} ({start} — {end}) ---")
    print(df_regime.to_string())

    # Сохраняем CSV
    fname = f"{OUT_DIR}/regime_{regime_key}.csv"
    df_regime.to_csv(fname)
    print(f"Saved: {fname}")

print("\n=== ВСЕ ФАЙЛЫ СОХРАНЕНЫ ===")
print(f"Папка: {OUT_DIR}/")