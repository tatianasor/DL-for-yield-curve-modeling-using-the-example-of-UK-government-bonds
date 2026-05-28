import pandas as pd

# загрузка обработанных данных
df = pd.read_csv("UK_yield_curve_daily_processed.csv")

print("\n========== BASIC INFO ==========")
print(df.info())

print("\n========== FIRST ROWS ==========")
print(df.head())

print("\n========== LAST ROWS ==========")
print(df.tail())

# -----------------------------
# проверка пропусков
# -----------------------------

print("\n========== MISSING VALUES ==========")
missing_total = df.isna().sum().sum()
print("Total missing values:", missing_total)

print("\nMissing by column:")
print(df.isna().sum())

# -----------------------------
# проверка типов данных
# -----------------------------

print("\n========== DATA TYPES ==========")
print(df.dtypes)

# -----------------------------
# проверка дат
# -----------------------------

print("\n========== DATE CHECK ==========")

df["Date"] = pd.to_datetime(df["Date"])

print("Start date:", df["Date"].min())
print("End date:", df["Date"].max())
print("Number of observations:", len(df))

# -----------------------------
# проверка maturities
# -----------------------------

print("\n========== MATURITY COLUMNS ==========")

maturities = df.columns[1:]
print("Number of maturities:", len(maturities))

print("First maturities:", maturities[:5])
print("Last maturities:", maturities[-5:])

# -----------------------------
# статистика доходностей
# -----------------------------

print("\n========== YIELD STATISTICS ==========")

print(df.describe())

# -----------------------------
# проверка диапазона доходностей
# -----------------------------

print("\n========== RANGE CHECK ==========")

print("Minimum yield:", df.iloc[:,1:].min().min())
print("Maximum yield:", df.iloc[:,1:].max().max())

# -----------------------------
# проверка монотонности maturities
# -----------------------------

print("\n========== MATURITY ORDER CHECK ==========")

maturity_values = [float(col) for col in maturities]

if maturity_values == sorted(maturity_values):
    print("Maturities are correctly sorted")
else:
    print("WARNING: maturities not sorted")

print("\n========== DATA CHECK COMPLETE ==========")

import matplotlib.pyplot as plt

row = df.iloc[-1,1:]

plt.plot(row.index.astype(float), row.values)
plt.xlabel("Maturity")
plt.ylabel("Yield")
plt.title("UK Yield Curve")
plt.show()