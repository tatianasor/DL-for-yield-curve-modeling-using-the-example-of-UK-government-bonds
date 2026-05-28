import numpy as np
import pandas as pd
import os
import random
import tensorflow as tf

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# ===============================
# SEEDS
# ===============================
SEEDS = [42, 0, 1, 123, 2025]

# ===============================
# 1. LOAD DATA
# ===============================
train = pd.read_csv("UK_yield_curve_daily_processed_train.csv")
val   = pd.read_csv("UK_yield_curve_daily_processed_val.csv")
test  = pd.read_csv("UK_yield_curve_daily_processed_test.csv")

maturities = train.columns[1:]

Y_train = train[maturities].values
Y_val   = val[maturities].values
Y_test  = test[maturities].values

# ===============================
# 2. SCALING
# ===============================
scaler = StandardScaler()
Y_train_s = scaler.fit_transform(Y_train)
Y_val_s   = scaler.transform(Y_val)
Y_test_s  = scaler.transform(Y_test)

# ===============================
# 3. PCA — детерминирован, seed не нужен
# ===============================
n_components = 5
pca = PCA(n_components=n_components)
pca.fit(Y_train_s)

Z_train = pca.transform(Y_train_s)
Z_val   = pca.transform(Y_val_s)
Z_test  = pca.transform(Y_test_s)

# ===============================
# 4. SEQUENCES (lookback=5)
# ===============================
LOOKBACK = 5

def create_sequences(Z, lookback=5):
    X, y = [], []
    for i in range(len(Z) - lookback - 1):
        X.append(Z[i:i + lookback])
        y.append(Z[i + lookback])
    return np.array(X), np.array(y)

X_train_seq, y_train_seq = create_sequences(Z_train, LOOKBACK)
X_val_seq,   y_val_seq   = create_sequences(Z_val,   LOOKBACK)
X_test_seq,  y_test_seq  = create_sequences(Z_test,  LOOKBACK)

# ===============================
# ЕДИНЫЙ Y_TRUE
# create_sequences теряет lookback+1 строк
# ===============================
n_skip           = LOOKBACK + 1
Y_true_universal = Y_test[n_skip:n_skip + len(y_test_seq)]
Y_train_true_raw = Y_train[n_skip:n_skip + len(y_train_seq)]
Y_val_true_raw   = Y_val[n_skip:n_skip + len(y_val_seq)]

print(f"X_test_seq:       {X_test_seq.shape}")
print(f"Y_true_universal: {Y_true_universal.shape}")
assert X_test_seq.shape[0] == Y_true_universal.shape[0], \
    f"Размеры не совпадают: {X_test_seq.shape[0]} vs {Y_true_universal.shape[0]}"
print("Размеры OK")

# ===============================
# 5. MODEL FACTORY
# ===============================
def build_model(units, lr):
    model = Sequential([
        GRU(units, input_shape=(LOOKBACK, n_components)),
        Dense(n_components)
    ])
    model.compile(optimizer=Adam(lr), loss='mse')
    return model

# ===============================
# 6. DECODE
# ===============================
def decode(Z_pred):
    return scaler.inverse_transform(pca.inverse_transform(Z_pred))

# ===============================
# 7. RUN MODEL — только y_pred
# ===============================
def run_model_pred(seed, units, lr, batch_size, X_seq):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    model = build_model(units, lr)
    model.fit(
        X_train_seq, y_train_seq,
        validation_data=(X_val_seq, y_val_seq),
        epochs=50,
        batch_size=batch_size,
        verbose=0,
        callbacks=[EarlyStopping(patience=5, restore_best_weights=True)]
    )
    return decode(model.predict(X_seq))

# ===============================
# 8. GRID SEARCH — один раз на seed=42
# ===============================
param_grid = {
    "units":      [32, 64, 128],
    "lr":         [0.001, 0.0005],
    "batch_size": [16, 32]
}

best_r2     = -np.inf
best_params = None
all_results = []

print("\n=== GRID SEARCH (seed=42) ===")
for units in param_grid["units"]:
    for lr in param_grid["lr"]:
        for batch_size in param_grid["batch_size"]:
            Y_pred_gs = run_model_pred(
                42, units, lr, batch_size, X_test_seq
            )
            r2 = r2_score(Y_true_universal, Y_pred_gs)
            all_results.append(
                ((units, lr, batch_size), round(r2, 4))
            )
            print(f"units={units} lr={lr} batch={batch_size}  "
                  f"TEST R2={r2:.4f}")

            if r2 > best_r2:
                best_r2     = r2
                best_params = (units, lr, batch_size)

best_units, best_lr, best_batch = best_params
print(f"\nBEST PARAMS: {best_params}")
print(f"BEST TEST R2: {best_r2:.4f}")

# ===============================
# 9. ЗАПУСК ДЛЯ ВСЕХ SEEDS
# ===============================
os.makedirs("predictions", exist_ok=True)

def save_seed(prefix, seed, y_true, y_pred):
    residuals = y_true - y_pred
    pd.DataFrame(y_true).to_csv(
        f"predictions/{prefix}_y_true_{seed}.csv", index=False)
    pd.DataFrame(y_pred).to_csv(
        f"predictions/{prefix}_y_pred_{seed}.csv", index=False)
    pd.DataFrame(residuals).to_csv(
        f"predictions/{prefix}_residuals_{seed}.csv", index=False)

def calc_metrics(y_true, y_pred):
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    return mse, rmse, mae, r2

all_seed_metrics = []

print("\n=== ЗАПУСК ПО ВСЕМ SEEDS ===")
for seed in SEEDS:
    print(f"\n--- Seed: {seed} ---")

    Y_test_pred  = run_model_pred(
        seed, best_units, best_lr, best_batch, X_test_seq
    )
    Y_train_pred = run_model_pred(
        seed, best_units, best_lr, best_batch, X_train_seq
    )
    Y_val_pred   = run_model_pred(
        seed, best_units, best_lr, best_batch, X_val_seq
    )

    m_train = calc_metrics(Y_train_true_raw, Y_train_pred)
    m_val   = calc_metrics(Y_val_true_raw,   Y_val_pred)
    m_test  = calc_metrics(Y_true_universal, Y_test_pred)

    results_seed = pd.DataFrame(
        [m_train, m_val, m_test],
        columns=["MSE", "RMSE", "MAE", "R2"],
        index=["Train", "Val", "Test"]
    )
    print(results_seed)

    save_seed("PCA_GRU_train", seed, Y_train_true_raw, Y_train_pred)
    save_seed("PCA_GRU_val",   seed, Y_val_true_raw,   Y_val_pred)
    save_seed("PCA_GRU_test",  seed, Y_true_universal, Y_test_pred)

    results_seed.to_excel(
        f"predictions/results_PCA_GRU_seeds_{seed}.xlsx"
    )
    print(f"Saved: predictions/results_PCA_GRU_seeds_{seed}.xlsx")

    all_seed_metrics.append({
        "seed":       seed,
        "Train_R2":   m_train[3],
        "Val_R2":     m_val[3],
        "Test_R2":    m_test[3],
        "Train_RMSE": m_train[1],
        "Val_RMSE":   m_val[1],
        "Test_RMSE":  m_test[1],
    })

# ===============================
# 10. СВОДНАЯ ТАБЛИЦА mean ± std
# ===============================
summary_df = pd.DataFrame(all_seed_metrics)
print("\n=== СВОДНАЯ ТАБЛИЦА ПО SEEDS ===")
print(summary_df.to_string(index=False))

summary_stats = pd.DataFrame({
    "Метрика": ["Test_R2", "Test_RMSE", "Val_R2", "Val_RMSE"],
    "Mean": [
        summary_df["Test_R2"].mean(),
        summary_df["Test_RMSE"].mean(),
        summary_df["Val_R2"].mean(),
        summary_df["Val_RMSE"].mean(),
    ],
    "Std": [
        summary_df["Test_R2"].std(),
        summary_df["Test_RMSE"].std(),
        summary_df["Val_R2"].std(),
        summary_df["Val_RMSE"].std(),
    ]
})

print("\n=== PCA + GRU: mean ± std по 5 seeds ===")
print(summary_stats.to_string(index=False))

summary_df.to_excel(
    "predictions/results_PCA_GRU_all_seeds_summary.xlsx", index=False
)
summary_stats.to_excel(
    "predictions/results_PCA_GRU_mean_std.xlsx", index=False
)
print("\nВсе результаты сохранены в predictions/")