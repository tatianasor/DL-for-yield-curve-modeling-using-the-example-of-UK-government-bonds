import pandas as pd

# -----------------------------
# 1. Загрузка PCA-файлов
# -----------------------------
train_pca = pd.read_csv("UK_yield_curve_PCA_train.csv")
val_pca = pd.read_csv("UK_yield_curve_PCA_val.csv")
test_pca = pd.read_csv("UK_yield_curve_PCA_test.csv")

pca_features = ["PC1", "PC2", "PC3"]
rolling_windows = [3,6,12]

def create_lstm_features(df, pca_features, rolling_windows=[3,6,12]):
    df_feat = df.copy()

    # -----------------------------
    # 1. ΔPC (разности)
    # -----------------------------
    for pc in pca_features:
        df_feat[f"d{pc}"] = df_feat[pc].diff()
    df_feat.fillna(0, inplace=True)

    # -----------------------------
    # 2. Скользящие средние с разными окнами
    # -----------------------------
    for window in rolling_windows:
        for pc in pca_features:
            df_feat[f"MA{window}_{pc}"] = df_feat[pc].rolling(window).mean()
    df_feat = df_feat.bfill()

    # -----------------------------
    # 3. Финальный список фич
    # -----------------------------
    feature_cols = (
        pca_features +
        [f"d{pc}" for pc in pca_features] +
        [f"MA{window}_{pc}" for window in rolling_windows for pc in pca_features]
    )
    df_feat_final = df_feat[["Date"] + feature_cols]

    return df_feat_final


# -----------------------------
# 2. Создаем фичи
# -----------------------------
train_lstm = create_lstm_features(train_pca, pca_features, rolling_windows)
val_lstm = create_lstm_features(val_pca, pca_features, rolling_windows)
test_lstm = create_lstm_features(test_pca, pca_features, rolling_windows)

# -----------------------------
# 3. Сохраняем
# -----------------------------
train_lstm.to_csv("UK_yield_curve_LSTM_train.csv", index=False)
val_lstm.to_csv("UK_yield_curve_LSTM_val.csv", index=False)
test_lstm.to_csv("UK_yield_curve_LSTM_test.csv", index=False)

print("LSTM feature datasets saved:")
print("LSTM feature datasets saved:")
print(" - UK_yield_curve_LSTM_train.csv")
print(" - UK_yield_curve_LSTM_val.csv")
print(" - UK_yield_curve_LSTM_test.csv")