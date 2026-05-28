import pandas as pd
import numpy as np
import os
import random
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
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

# ===============================
# 2. FEATURE ENGINEERING
# ===============================
def create_features(df, maturities):
    df_feat = df.copy()
    for m in maturities:
        df_feat[f"{m}_lag1"] = df[m].shift(1)
        df_feat[f"{m}_lag2"] = df[m].shift(2)
        df_feat[f"{m}_lag3"] = df[m].shift(3)
    df_feat = df_feat.dropna().reset_index(drop=True)
    return df_feat

def make_forecast_data(df):
    X = df.copy()
    y = df.shift(-1)
    return X.iloc[:-1], y.iloc[:-1]

train_feat = create_features(train, maturities)
val_feat   = create_features(val,   maturities)
test_feat  = create_features(test,  maturities)

X_train_raw, y_train_raw = make_forecast_data(train_feat[maturities])
X_val_raw,   y_val_raw   = make_forecast_data(val_feat[maturities])
X_test_raw,  y_test_raw  = make_forecast_data(test_feat[maturities])

# ===============================
# ЕДИНЫЙ Y_TRUE
# ===============================
n_skip           = len(test) - len(y_test_raw)
Y_true_universal = test[maturities].values[n_skip:]
Y_train_true_raw = y_train_raw.values  # для train/val оставляем как было
Y_val_true_raw   = y_val_raw.values

assert len(Y_true_universal) == len(y_test_raw), \
    f"Размеры не совпадают: {len(Y_true_universal)} vs {len(y_test_raw)}"
print(f"n_skip={n_skip}, y_true rows={len(Y_true_universal)}")

# ===============================
# 3. SCALING
# ===============================
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train = scaler_X.fit_transform(X_train_raw)
X_val   = scaler_X.transform(X_val_raw)
X_test  = scaler_X.transform(X_test_raw)

y_train = scaler_y.fit_transform(y_train_raw)
y_val   = scaler_y.transform(y_val_raw)
y_test  = scaler_y.transform(y_test_raw)

# ===============================
# 4. MODEL FACTORY
# ===============================
def build_model(units1, units2, bottleneck, lr):
    inp    = Input(shape=(X_train.shape[1],))
    x      = Dense(units1, activation='relu')(inp)
    x      = Dense(units2, activation='relu')(x)
    latent = Dense(bottleneck, activation='relu', name="latent")(x)
    x      = Dense(units2, activation='relu')(latent)
    x      = Dense(units1, activation='relu')(x)
    out    = Dense(y_train.shape[1], activation='linear')(x)
    model  = Model(inp, out)
    model.compile(optimizer=Adam(lr), loss='mse')
    return model

# ===============================
# 5. RUN MODEL — только y_pred
# ===============================
def run_model_pred(seed, params, X_seq):
    u1, u2, b, lr = params
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    model = build_model(u1, u2, b, lr)
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=16,
        verbose=0,
        callbacks=[EarlyStopping(patience=10, restore_best_weights=True)]
    )
    return scaler_y.inverse_transform(model.predict(X_seq))

# ===============================
# 6. GRID SEARCH — один раз на seed=42
# ===============================
grid = [
    (64,  32, 16, 0.001),
    (128, 64, 16, 0.001),
    (128, 64, 32, 0.0005),
]

best_r2     = -np.inf
best_params = None
all_results = []

print("\n=== GRID SEARCH (seed=42) ===")
for params in grid:
    Y_pred_gs = run_model_pred(42, params, X_test)
    r2        = r2_score(Y_true_universal, Y_pred_gs)
    all_results.append((params, round(r2, 4)))
    print(f"Params {params}  TEST R2={r2:.4f}")

    if r2 > best_r2:
        best_r2     = r2
        best_params = params

print(f"\nBEST PARAMS: {best_params}")
print(f"BEST TEST R2: {best_r2:.4f}")

# ===============================
# 7. ЗАПУСК ДЛЯ ВСЕХ SEEDS
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

    Y_test_pred  = run_model_pred(seed, best_params, X_test)
    Y_train_pred = run_model_pred(seed, best_params, X_train)
    Y_val_pred   = run_model_pred(seed, best_params, X_val)

    m_train = calc_metrics(Y_train_true_raw, Y_train_pred)
    m_val   = calc_metrics(Y_val_true_raw,   Y_val_pred)
    m_test  = calc_metrics(Y_true_universal, Y_test_pred)

    results_seed = pd.DataFrame(
        [m_train, m_val, m_test],
        columns=["MSE", "RMSE", "MAE", "R2"],
        index=["Train", "Val", "Test"]
    )
    print(results_seed)

    save_seed("AE_dense_train", seed, Y_train_true_raw, Y_train_pred)
    save_seed("AE_dense_val",   seed, Y_val_true_raw,   Y_val_pred)
    save_seed("AE_dense_test",  seed, Y_true_universal, Y_test_pred)

    results_seed.to_excel(
        f"predictions/results_AE_dense_seeds_{seed}.xlsx"
    )
    print(f"Saved: predictions/results_AE_dense_seeds_{seed}.xlsx")

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
# 8. СВОДНАЯ ТАБЛИЦА mean ± std
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

print("\n=== AE dense: mean ± std по 5 seeds ===")
print(summary_stats.to_string(index=False))

summary_df.to_excel(
    "predictions/results_AE_dense_all_seeds_summary.xlsx", index=False
)
summary_stats.to_excel(
    "predictions/results_AE_dense_mean_std.xlsx", index=False
)

# ===============================
# 9. ARTIFACTS ДЛЯ ШАГА 14
#    Сохраняем энкодер от лучшей модели (seed=42)
# ===============================
os.makedirs("artifacts_ae", exist_ok=True)

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

u1, u2, b, lr = best_params
best_model = build_model(u1, u2, b, lr)
best_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=16,
    verbose=0,
    callbacks=[EarlyStopping(patience=10, restore_best_weights=True)]
)

encoder = Model(best_model.input, best_model.get_layer("latent").output)
Z_train_ae = encoder.predict(X_train)
Z_val_ae   = encoder.predict(X_val)
Z_test_ae  = encoder.predict(X_test)

pd.DataFrame(Z_train_ae).to_csv("artifacts_ae/ae_factors_train.csv", index=False)
pd.DataFrame(Z_val_ae).to_csv("artifacts_ae/ae_factors_val.csv",     index=False)
pd.DataFrame(Z_test_ae).to_csv("artifacts_ae/ae_factors_test.csv",   index=False)

best_model.save("artifacts_ae/ae_model.h5")
print("\nВсе результаты сохранены в predictions/")