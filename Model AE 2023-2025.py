import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

import joblib

# -----------------------------
# CREATE OUTPUT FOLDER
# -----------------------------
os.makedirs("predictions", exist_ok=True)

MODEL_NAME = "AE_2023-2025"

# -----------------------------
# 1. LOAD DATA
# -----------------------------
train = pd.read_csv("UK_yield_curve_daily_processed_2023-2025_train.csv")
val = pd.read_csv("UK_yield_curve_daily_processed_2023-2025_val.csv")
test = pd.read_csv("UK_yield_curve_daily_processed_2023-2025_test.csv")

maturities = train.columns[1:]

# =========================================================
# 2. FORECAST SETUP (t → t+1)
# =========================================================
def make_forecast_data(df):
    X = df.copy()
    y = df.shift(-1)

    X = X.iloc[:-1]
    y = y.iloc[:-1]
    return X, y


# =========================================================
# 3. LAG FEATURES (no leakage)
# =========================================================
def create_features(df, maturities):
    df_feat = df.copy()
    new_cols = {}

    for m in maturities:
        new_cols[f"{m}_lag1"] = df[m].shift(1)
        new_cols[f"{m}_lag2"] = df[m].shift(2)
        new_cols[f"{m}_lag3"] = df[m].shift(3)

    df_feat = pd.concat([df_feat, pd.DataFrame(new_cols)], axis=1)
    df_feat = df_feat.dropna().reset_index(drop=True)

    return df_feat


# =========================================================
# 🔥 IMPORTANT: ENSURE NO LEAKAGE FROM FUTURE REGIME
# =========================================================
train = train.copy()
val = val.copy()

# (уже режим до 2023-2025 — но оставляем безопасность)
train_feat = create_features(train, maturities)
val_feat = create_features(val, maturities)
test_feat = create_features(test, maturities)

# -----------------------------
# ALIGN TARGETS
# -----------------------------
X_train_raw, y_train_raw = make_forecast_data(train_feat[maturities])
X_val_raw, y_val_raw = make_forecast_data(val_feat[maturities])
X_test_raw, y_test_raw = make_forecast_data(test_feat[maturities])


# =========================================================
# 4. SCALING
# =========================================================
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train = scaler_X.fit_transform(X_train_raw)
X_val = scaler_X.transform(X_val_raw)
X_test = scaler_X.transform(X_test_raw)

y_train = scaler_y.fit_transform(y_train_raw)
y_val = scaler_y.transform(y_val_raw)
y_test = scaler_y.transform(y_test_raw)


# =========================================================
# 5. MODEL
# =========================================================
input_dim = X_train.shape[1]
output_dim = y_train.shape[1]

inp = Input(shape=(input_dim,))
x = Dense(64, activation='relu')(inp)
x = Dense(32, activation='relu')(x)
bottleneck = Dense(16, activation='relu')(x)
x = Dense(32, activation='relu')(bottleneck)
x = Dense(64, activation='relu')(x)
out = Dense(output_dim, activation='linear')(x)

model = Model(inp, out)
model.compile(optimizer=Adam(0.001), loss='mse')


# =========================================================
# 6. TRAIN
# =========================================================
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=16,
    verbose=2,
    callbacks=[early_stop]
)


# =========================================================
# 7. METRICS
# =========================================================
def metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, mae, r2


# =========================================================
# 8. SAVE FUNCTION
# =========================================================
def save_predictions(name, y_true, y_pred):
    pd.DataFrame(y_true, columns=maturities).to_csv(
        f"predictions/{MODEL_NAME}_{name}_y_true.csv", index=False
    )
    pd.DataFrame(y_pred, columns=maturities).to_csv(
        f"predictions/{MODEL_NAME}_{name}_y_pred.csv", index=False
    )
    pd.DataFrame(y_true - y_pred, columns=maturities).to_csv(
        f"predictions/{MODEL_NAME}_{name}_residuals.csv", index=False
    )


# =========================================================
# 9. STATIC
# =========================================================
train_pred = scaler_y.inverse_transform(model.predict(X_train))
val_pred = scaler_y.inverse_transform(model.predict(X_val))
test_pred = scaler_y.inverse_transform(model.predict(X_test))

static_df = pd.DataFrame(
    [
        metrics(y_train_raw, train_pred),
        metrics(y_val_raw, val_pred),
        metrics(y_test_raw, test_pred)
    ],
    columns=["MSE", "RMSE", "MAE", "R2"],
    index=["Train", "Val", "Test"]
)

print("\n=== STATIC FORECAST (2023-2025 REGIME) ===")
print(static_df)

# =========================================================
# 🔥 FULL REQUIRED SAVES FOR STAT TESTS
# =========================================================
save_predictions("train_static", y_train_raw.values, train_pred)
save_predictions("val_static", y_val_raw.values, val_pred)
save_predictions("test_static", y_test_raw.values, test_pred)


# =========================================================
# 10. ROLLING WINDOW
# =========================================================
def rolling_window(X, y, window=50):
    preds, trues = [], []

    for i in range(window, len(X)):
        pred = model.predict(X[i-window:i], verbose=0)[-1]
        preds.append(pred)
        trues.append(y[i])

    preds = scaler_y.inverse_transform(np.array(preds))
    trues = np.array(trues)
    return preds, trues, metrics(trues, preds)


tr_pred, tr_true, tr_m = rolling_window(X_train, y_train_raw.values)
va_pred, va_true, va_m = rolling_window(X_val, y_val_raw.values)
te_pred, te_true, te_m = rolling_window(X_test, y_test_raw.values)


rolling_df = pd.DataFrame(
    [tr_m, va_m, te_m],
    columns=["MSE", "RMSE", "MAE", "R2"],
    index=["Train", "Val", "Test"]
)

print("\n=== ROLLING WINDOW FORECAST ===")
print(rolling_df)

save_predictions("train_rolling", tr_true, tr_pred)
save_predictions("val_rolling", va_true, va_pred)
save_predictions("test_rolling", te_true, te_pred)


# =========================================================
# 11. EXPANDING WINDOW
# =========================================================
def expanding_window(X, y, min_window=50):
    preds, trues = [], []

    for i in range(min_window, len(X)):
        pred = model.predict(X[:i], verbose=0)[-1]
        preds.append(pred)
        trues.append(y[i])

    preds = scaler_y.inverse_transform(np.array(preds))
    trues = np.array(trues)
    return preds, trues, metrics(trues, preds)


tr_pred, tr_true, tr_m = expanding_window(X_train, y_train_raw.values)
va_pred, va_true, va_m = expanding_window(X_val, y_val_raw.values)
te_pred, te_true, te_m = expanding_window(X_test, y_test_raw.values)


expanding_df = pd.DataFrame(
    [tr_m, va_m, te_m],
    columns=["MSE", "RMSE", "MAE", "R2"],
    index=["Train", "Val", "Test"]
)

print("\n=== EXPANDING WINDOW FORECAST ===")
print(expanding_df)

save_predictions("train_expanding", tr_true, tr_pred)
save_predictions("val_expanding", va_true, va_pred)
save_predictions("test_expanding", te_true, te_pred)


# =========================================================
# 12. SAVE MODEL
# =========================================================
joblib.dump(model, f"{MODEL_NAME}_model.pkl")
joblib.dump(scaler_X, f"{MODEL_NAME}_X_scaler.pkl")
joblib.dump(scaler_y, f"{MODEL_NAME}_y_scaler.pkl")

print("\nDONE: AE_2023-2025 full pipeline saved")