import pandas as pd
from pathlib import Path

# =========================
# 1. Load data
# =========================

df = pd.read_csv("UK_yield_curve_daily_processed.csv")

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

print("Total observations:", len(df))


# =========================
# 2. Global split (70/15/15)
# =========================

train_size = int(len(df) * 0.70)
val_size = int(len(df) * 0.15)

train_end = train_size
val_end = train_size + val_size

train_df = df.iloc[:train_end].reset_index(drop=True)
val_df   = df.iloc[train_end:val_end].reset_index(drop=True)
test_df  = df.iloc[val_end:].reset_index(drop=True)


# =========================
# 3. Save global splits
# =========================

train_df.to_csv("UK_yield_curve_daily_processed_train.csv", index=False)
val_df.to_csv("UK_yield_curve_daily_processed_val.csv", index=False)
test_df.to_csv("UK_yield_curve_daily_processed_test.csv", index=False)


# =========================
# 4. Define regime function
# =========================

def split_by_regime(df, start, end):
    return df[(df["Date"] >= start) & (df["Date"] <= end)].copy()


def save_regime(df, name):
    df = df.reset_index(drop=True)

    train_size = int(len(df) * 0.70)
    val_size = int(len(df) * 0.15)

    train = df.iloc[:train_size]
    val = df.iloc[train_size:train_size + val_size]
    test = df.iloc[train_size + val_size:]

    train.to_csv(f"UK_yield_curve_daily_processed_{name}_train.csv", index=False)
    val.to_csv(f"UK_yield_curve_daily_processed_{name}_val.csv", index=False)
    test.to_csv(f"UK_yield_curve_daily_processed_{name}_test.csv", index=False)

    print(f"\n{name} saved:",
          len(train), len(val), len(test))


# =========================
# 5. Define regimes
# =========================

regimes = {
    "2019": ("2019-01-01", "2019-12-31"),
    "2020-2021": ("2020-01-01", "2021-12-31"),
    "2022": ("2022-01-01", "2022-12-31"),
    "2023-2025": ("2023-01-01", "2025-12-31")
}


# =========================
# 6. Create regime datasets
# =========================

for name, (start, end) in regimes.items():
    regime_df = split_by_regime(df, start, end)
    save_regime(regime_df, name)


print("\nALL FILES SAVED (15 datasets total)")