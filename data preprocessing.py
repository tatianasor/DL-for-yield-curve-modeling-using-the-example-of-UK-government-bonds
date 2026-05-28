import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# -----------------------------
# 1. Load
# -----------------------------
df = pd.read_csv("UK_yield_curve daily.csv")

df = df.rename(columns={df.columns[0]: "Date"})
df["Date"] = pd.to_datetime(df["Date"])

# -----------------------------
# 2. maturities
# -----------------------------
maturity_cols = df.columns[1:]
df.columns = ["Date"] + [float(c) for c in maturity_cols]

maturities = sorted(df.columns[1:])
df = df[["Date"] + maturities]

# -----------------------------
# 3. CRITICAL FIX: time index
# -----------------------------
df = df.sort_values("Date")
df = df.set_index("Date")

# 👉 make real daily grid
df = df.asfreq("D")

print("Missing after reindex:", df.isna().sum().sum())

# -----------------------------
# 4. Fill missing values correctly
# -----------------------------

# time-aware interpolation (VERY IMPORTANT)
df[maturities] = df[maturities].interpolate(method="time")

# fallback fills
df[maturities] = df[maturities].ffill()
df[maturities] = df[maturities].bfill()

# -----------------------------
# 5. back to normal
# -----------------------------
df = df.reset_index()

# -----------------------------
# 6. scaling
# -----------------------------
scaler = StandardScaler()
df_scaled = df.copy()

df_scaled[maturities] = scaler.fit_transform(df_scaled[maturities])

# -----------------------------
# 7. check
# -----------------------------
print("Final missing:", df_scaled[maturities].isna().sum().sum())
print(df_scaled.head())

# -----------------------------
# 8. save
# -----------------------------
df_scaled.to_csv("UK_yield_curve_daily_processed.csv", index=False)
print("Saved corrected dataset")