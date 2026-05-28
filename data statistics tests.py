"""
Шаг 8. Статистические тесты значимости:
    1) Тест Diebold-Mariano (попарное сравнение моделей по точности прогноза).
    2) Bootstrap-доверительные интервалы для Test R^2 (B = 1000).
    3) Сводка mean +/- std по 5 seeds для всех нейросетевых моделей.

Сохраняет результаты в predictions/stat_tests/.
"""

import numpy as np
import pandas as pd
import os
from scipy.stats import norm
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# =========================================================
# PATH
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PRED_DIR = os.path.join(BASE_DIR, "predictions")
OUT_DIR  = os.path.join(PRED_DIR, "stat_tests")
os.makedirs(OUT_DIR, exist_ok=True)

print("Predictions dir:", PRED_DIR)
print("Exists:", os.path.exists(PRED_DIR))
print("Output dir:", OUT_DIR)

# =========================================================
# КОНФИГУРАЦИЯ
# =========================================================
SEEDS = [42, 0, 1, 123, 2025]

# Лучшие seeds для отчёта по DM и bootstrap
best_seeds = {
    "PCA_LR":   None,       # детерминирован
    "AE_dense": 42,
    "PCA_MLP":  42,
    "AE_MLP":   123,
    "PCA_LSTM": 1,
    "AE_LSTM":  123,
    "PCA_GRU":  1,
    "AE_GRU":   2025,
}

# Для mean +/- std по seeds — только стохастические модели
stochastic_models = [
    "AE_dense", "PCA_MLP", "AE_MLP",
    "PCA_LSTM", "AE_LSTM", "PCA_GRU", "AE_GRU"
]

# =========================================================
# LOADERS
# =========================================================
def load_matrix(file):
    path = os.path.join(PRED_DIR, file)
    return pd.read_csv(path).values

def load_pred(model, seed):
    if seed is None:
        fname = f"{model}_test_y_pred.csv"
    else:
        fname = f"{model}_test_y_pred_{seed}.csv"
    return load_matrix(fname)

def load_true(model, seed):
    if seed is None:
        fname = f"{model}_test_y_true.csv"
    else:
        fname = f"{model}_test_y_true_{seed}.csv"
    return load_matrix(fname)

# =========================================================
# 1) ЗАГРУЗКА ДЛЯ DM-ТЕСТА И BOOTSTRAP (лучшие seeds)
# =========================================================
y_true_mat = load_matrix("PCA_LR_test_y_true.csv")
print(f"\ny_true shape: {y_true_mat.shape}")

preds_mat = {}
for model, seed in best_seeds.items():
    try:
        y_pred = load_pred(model, seed)
        preds_mat[model] = y_pred
        print(f"OK  {model:10s} (seed={seed}): {y_pred.shape}")
    except FileNotFoundError as e:
        print(f"ERR {model}: файл не найден — {e}")

# Выравниваем длины (GRU теряет lookback+1 = 6 строк по сравнению с PCA+LR)
min_len = min([len(y_true_mat)] + [len(v) for v in preds_mat.values()])
diff = len(y_true_mat) - min_len
# Обрезаем СНАЧАЛА (берём последние min_len строк), чтобы все модели
# смотрели на ту же тестовую подвыборку, что и GRU
y_true_aligned = y_true_mat[diff:]
preds_aligned = {}
for k, v in preds_mat.items():
    cut = len(v) - min_len
    preds_aligned[k] = v[cut:]
print(f"\nПосле выравнивания: {min_len} наблюдений")

# =========================================================
# 2) DM-ТЕСТ
# =========================================================
def row_mse(y_true, y_pred):
    """MSE по строкам (по тенорам) -> вектор длины T."""
    return np.mean((y_true - y_pred) ** 2, axis=1)

def dm_test(e1, e2):
    """
    Diebold-Mariano для квадратичной функции потерь.
    e1, e2 — векторы row_mse двух моделей.

    H0: модели одинаково точны.
    DM > 0  -> первая модель ХУЖЕ (её ошибка больше).
    DM < 0  -> первая модель ЛУЧШЕ.
    """
    d = e1 - e2
    mean_d = np.mean(d)
    var_d  = np.var(d, ddof=1)
    if var_d < 1e-12:
        return 0.0, 1.0
    dm_stat = mean_d / np.sqrt(var_d / len(d))
    p_value = 2 * (1 - norm.cdf(abs(dm_stat)))
    return dm_stat, max(p_value, 1e-16)

errors = {name: row_mse(y_true_aligned, pred)
          for name, pred in preds_aligned.items()}

model_names = list(errors.keys())
n_models = len(model_names)

dm_stat_mat = pd.DataFrame(np.zeros((n_models, n_models)),
                           index=model_names, columns=model_names)
dm_pval_mat = pd.DataFrame(np.zeros((n_models, n_models)),
                           index=model_names, columns=model_names)

for i, m1 in enumerate(model_names):
    for j, m2 in enumerate(model_names):
        if i == j:
            dm_stat_mat.iloc[i, j] = 0.0
            dm_pval_mat.iloc[i, j] = 1.0
        else:
            stat, p = dm_test(errors[m1], errors[m2])
            dm_stat_mat.iloc[i, j] = stat
            dm_pval_mat.iloc[i, j] = p

print("\n=== DM-СТАТИСТИКИ (строка vs столбец) ===")
print(dm_stat_mat.round(3))
print("\n=== DM P-VALUES ===")
print(dm_pval_mat.round(4))

dm_stat_mat.to_excel(os.path.join(OUT_DIR, "dm_statistics.xlsx"))
dm_pval_mat.to_excel(os.path.join(OUT_DIR, "dm_pvalues.xlsx"))

# Сводка: для каждой модели — сколько других она статистически
# значимо обыгрывает (alpha = 0.05) и сколько ей проигрывает
alpha = 0.05
summary_rows = []
for m in model_names:
    n_better = 0   # модель m лучше другой (DM < 0, p < alpha)
    n_worse  = 0   # модель m хуже другой
    for other in model_names:
        if other == m:
            continue
        stat = dm_stat_mat.loc[m, other]
        p    = dm_pval_mat.loc[m, other]
        if p < alpha:
            if stat < 0:
                n_better += 1
            else:
                n_worse += 1
    summary_rows.append({
        "Модель":        m,
        "Лучше других":  n_better,
        "Хуже других":   n_worse,
        "Незначимо":     n_models - 1 - n_better - n_worse,
    })
dm_summary = pd.DataFrame(summary_rows)
print("\n=== DM-СВОДКА (alpha=0.05) ===")
print(dm_summary.to_string(index=False))
dm_summary.to_excel(os.path.join(OUT_DIR, "dm_summary.xlsx"), index=False)

# =========================================================
# 3) BOOTSTRAP CI ДЛЯ TEST R^2
# =========================================================
B = 1000
rng = np.random.default_rng(42)

def bootstrap_r2_ci(y_true, y_pred, B=1000, alpha=0.05, rng=None):
    """95%-й перцентильный bootstrap CI для R^2 на тестовой выборке."""
    if rng is None:
        rng = np.random.default_rng(42)
    T = len(y_true)
    r2_samples = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, T, size=T)
        r2_samples[b] = r2_score(y_true[idx], y_pred[idx])
    point = r2_score(y_true, y_pred)
    lo = np.quantile(r2_samples, alpha / 2)
    hi = np.quantile(r2_samples, 1 - alpha / 2)
    return point, lo, hi

print("\n=== BOOTSTRAP 95% CI ДЛЯ TEST R^2 (B=1000) ===")
ci_rows = []
for m in model_names:
    point, lo, hi = bootstrap_r2_ci(y_true_aligned, preds_aligned[m],
                                    B=B, rng=rng)
    ci_rows.append({
        "Модель":   m,
        "R2_point": point,
        "CI_low":   lo,
        "CI_high":  hi,
        "Width":    hi - lo,
    })
    print(f"{m:10s}  R2 = {point:.4f}  CI = [{lo:.4f}, {hi:.4f}]"
          f"  ширина = {hi - lo:.4f}")
ci_df = pd.DataFrame(ci_rows)
ci_df.to_excel(os.path.join(OUT_DIR, "bootstrap_r2_ci.xlsx"), index=False)

# =========================================================
# 4) MEAN +/- STD ПО 5 SEEDS
# =========================================================
print("\n=== MEAN +/- STD ПО 5 SEEDS ===")
seed_rows = []
for model in stochastic_models:
    r2_list, rmse_list, mae_list = [], [], []
    val_r2_list, val_rmse_list = [], []
    for seed in SEEDS:
        try:
            y_pred = load_pred(model, seed)
            y_true = load_true(model, seed)
            # выравнивание (на случай ручных правок)
            n = min(len(y_pred), len(y_true))
            y_pred = y_pred[-n:]
            y_true = y_true[-n:]

            r2_list.append(r2_score(y_true, y_pred))
            rmse_list.append(np.sqrt(mean_squared_error(y_true, y_pred)))
            mae_list.append(mean_absolute_error(y_true, y_pred))

            # val — из аналогичного файла, если есть
            try:
                yv_pred = load_matrix(f"{model}_val_y_pred_{seed}.csv")
                yv_true = load_matrix(f"{model}_val_y_true_{seed}.csv")
                nv = min(len(yv_pred), len(yv_true))
                val_r2_list.append(
                    r2_score(yv_true[-nv:], yv_pred[-nv:]))
                val_rmse_list.append(np.sqrt(
                    mean_squared_error(yv_true[-nv:], yv_pred[-nv:])))
            except FileNotFoundError:
                pass
        except FileNotFoundError:
            print(f"  пропуск: {model} seed={seed}")
            continue

    if len(r2_list) == 0:
        continue

    row = {
        "Модель":          model,
        "Запусков":        len(r2_list),
        "Test_R2_mean":    np.mean(r2_list),
        "Test_R2_std":     np.std(r2_list, ddof=1) if len(r2_list) > 1 else 0,
        "Test_RMSE_mean":  np.mean(rmse_list),
        "Test_RMSE_std":   np.std(rmse_list, ddof=1) if len(rmse_list) > 1 else 0,
        "Test_MAE_mean":   np.mean(mae_list),
        "Test_MAE_std":    np.std(mae_list, ddof=1) if len(mae_list) > 1 else 0,
        "Val_R2_mean":     np.mean(val_r2_list)   if val_r2_list else np.nan,
        "Val_R2_std":      (np.std(val_r2_list, ddof=1)
                            if len(val_r2_list) > 1 else 0),
        "Val_RMSE_mean":   np.mean(val_rmse_list) if val_rmse_list else np.nan,
        "Val_RMSE_std":    (np.std(val_rmse_list, ddof=1)
                            if len(val_rmse_list) > 1 else 0),
    }
    seed_rows.append(row)
    print(f"{model:10s}  Test R2 = {row['Test_R2_mean']:.4f} "
          f"+/- {row['Test_R2_std']:.4f}   "
          f"Test RMSE = {row['Test_RMSE_mean']:.5f} "
          f"+/- {row['Test_RMSE_std']:.5f}")

seed_df = pd.DataFrame(seed_rows)
seed_df.to_excel(os.path.join(OUT_DIR, "seeds_mean_std.xlsx"), index=False)

# =========================================================
# 5) ИТОГОВЫЙ СВОДНЫЙ ФАЙЛ
# =========================================================
final = ci_df.merge(
    seed_df[[
        "Модель", "Запусков",
        "Test_R2_mean", "Test_R2_std",
        "Test_RMSE_mean", "Test_RMSE_std"
    ]],
    on="Модель", how="left"
)
final.to_excel(os.path.join(OUT_DIR, "summary_all.xlsx"), index=False)

print(f"\nВсе результаты сохранены в {OUT_DIR}:")
print("  - dm_statistics.xlsx")
print("  - dm_pvalues.xlsx")
print("  - dm_summary.xlsx")
print("  - bootstrap_r2_ci.xlsx")
print("  - seeds_mean_std.xlsx")
print("  - summary_all.xlsx")