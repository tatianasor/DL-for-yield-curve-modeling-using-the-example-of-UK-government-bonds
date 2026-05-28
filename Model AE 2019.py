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
MODEL_NAME = "AE_2019_static"

# -----------------------------
# 1. LOAD DATA
# -----------------------------
train = pd.read_csv("UK_yield_curve_daily_processed_2019_train.csv")
val = pd.read_csv("UK_yield_curve_daily_processed_2019_val.csv")
test = pd.read_csv("UK_yield_curve_daily_processed_2019_test.csv")

maturities = train.columns[1:]

# =========================================================
# 2. CREATE LAGGED DATA FOR AR(3)
# =========================================================
def create_lagged_data(df, maturities, n_lags=3):
    """
    Создаёт X (лаги) и y (следующий день) для AR модели
    X: [lag1, lag2, lag3, ...] flattened
    y: t+1
    """
    X_list, y_list = [], []
    for i in range(n_lags, len(df)-1):
        # берем лаги t-3, t-2, t-1
        X_list.append(df[maturities].iloc[i-n_lags:i].values.flatten())
        # цель: t+1
        y_list.append(df[maturities].iloc[i+1].values)
    return np.array(X_list), np.array(y_list)

X_train_raw, y_train_raw = create_lagged_data(train, maturities)
X_val_raw, y_val_raw = create_lagged_data(val, maturities)
X_test_raw, y_test_raw = create_lagged_data(test, maturities)

# =========================================================
# 3. SCALING
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
# 4. MODEL
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
# 5. TRAIN
# =========================================================
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=16,
    verbose=2,
    callbacks=[early_stop]
)

# =========================================================
# 6. METRICS
# =========================================================
def metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, mae, r2

# =========================================================
# 7. SAVE FUNCTION
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
# 8. STATIC FORECAST
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

print("\n=== STATIC FORECAST (t → t+1) ===")
print(static_df)

# =========================================================
# 9. SAVE PREDICTIONS
# =========================================================
save_predictions("train_static", y_train_raw, train_pred)
save_predictions("val_static", y_val_raw, val_pred)
save_predictions("test_static", y_test_raw, test_pred)

# =========================================================
# 10. SAVE MODEL
# =========================================================
joblib.dump(model, f"{MODEL_NAME}_model.pkl")
joblib.dump(scaler_X, f"{MODEL_NAME}_X_scaler.pkl")
joblib.dump(scaler_y, f"{MODEL_NAME}_y_scaler.pkl")

print("\nDONE: AE_2019 static pipeline saved")