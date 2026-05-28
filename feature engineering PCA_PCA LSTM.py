import pandas as pd
from pathlib import Path

# =========================
# 1. Feature engineering
# =========================

def create_features(df, maturities, ma_windows=[3, 6, 12], lag_steps=[1, 2, 3]):

    df_feat = df.copy()
    new_cols = {}

    # -------------------------
    # 1. differences
    # -------------------------
    for m in maturities:
        new_cols[f"d_{m}"] = df[m].diff().fillna(0)

    # -------------------------
    # 2. lags
    # -------------------------
    for m in maturities:
        for lag in lag_steps:
            new_cols[f"{m}_lag{lag}"] = df[m].shift(lag).fillna(0)

    # -------------------------
    # 3. rolling stats
    # -------------------------
    for m in maturities:
        for w in ma_windows:
            new_cols[f"MA{w}_{m}"] = df[m].rolling(window=w, min_periods=1).mean()
            new_cols[f"VOL{w}_{m}"] = df[m].rolling(window=w, min_periods=1).std().fillna(0)

    # -------------------------
    # 4. slope feature
    # -------------------------
    short = maturities[:5]
    long = maturities[-5:]

    new_cols["curve_slope"] = df[long].mean(axis=1) - df[short].mean(axis=1)

    # -------------------------
    # merge
    # -------------------------
    df_feat = pd.concat([df_feat, pd.DataFrame(new_cols)], axis=1)
    df_feat.fillna(0, inplace=True)

    return df_feat


# =========================
# 2. maturities helper
# =========================

def get_maturities(df):
    return df.columns[1:]  # after Date


# =========================
# 3. process single file
# =========================

def process_file(input_path, output_path):
    df = pd.read_csv(input_path)
    df["Date"] = pd.to_datetime(df["Date"])

    maturities = get_maturities(df)

    df_feat = create_features(df, maturities)

    df_feat.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")


# =========================
# 4. file list
# =========================

files = [
    # GLOBAL
    ("UK_yield_curve_daily_processed_train.csv",
     "UK_yield_curve_daily_all_features_train.csv"),

    ("UK_yield_curve_daily_processed_val.csv",
     "UK_yield_curve_daily_all_features_val.csv"),

    ("UK_yield_curve_daily_processed_test.csv",
     "UK_yield_curve_daily_all_features_test.csv"),

    # 2019
    ("UK_yield_curve_daily_processed_2019_train.csv",
     "UK_yield_curve_daily_2019_features_train.csv"),

    ("UK_yield_curve_daily_processed_2019_val.csv",
     "UK_yield_curve_daily_2019_features_val.csv"),

    ("UK_yield_curve_daily_processed_2019_test.csv",
     "UK_yield_curve_daily_2019_features_test.csv"),

    # 2020-2021
    ("UK_yield_curve_daily_processed_2020-2021_train.csv",
     "UK_yield_curve_daily_2020-2021_features_train.csv"),

    ("UK_yield_curve_daily_processed_2020-2021_val.csv",
     "UK_yield_curve_daily_2020-2021_features_val.csv"),

    ("UK_yield_curve_daily_processed_2020-2021_test.csv",
     "UK_yield_curve_daily_2020-2021_features_test.csv"),

    # 2022
    ("UK_yield_curve_daily_processed_2022_train.csv",
     "UK_yield_curve_daily_2022_features_train.csv"),

    ("UK_yield_curve_daily_processed_2022_val.csv",
     "UK_yield_curve_daily_2022_features_val.csv"),

    ("UK_yield_curve_daily_processed_2022_test.csv",
     "UK_yield_curve_daily_2022_features_test.csv"),

    # 2023-2025
    ("UK_yield_curve_daily_processed_2023-2025_train.csv",
     "UK_yield_curve_daily_2023-2025_features_train.csv"),

    ("UK_yield_curve_daily_processed_2023-2025_val.csv",
     "UK_yield_curve_daily_2023-2025_features_val.csv"),

    ("UK_yield_curve_daily_processed_2023-2025_test.csv",
     "UK_yield_curve_daily_2023-2025_features_test.csv"),
]


# =========================
# 5. run all
# =========================

for inp, out in files:
    process_file(inp, out)

print("\nALL FEATURE FILES CREATED (15 files)")