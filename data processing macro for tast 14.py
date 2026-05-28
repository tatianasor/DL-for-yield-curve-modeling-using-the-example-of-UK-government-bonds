import pandas as pd
import numpy as np
import os

# ===============================
# ПУТИ
# ===============================
YIELD_PATH      = "UK_yield_curve_daily_processed.csv"
INFLATION_PATH  = "macro_inflation.csv"
BANK_RATE_XLSX  = "macro_Bank Rate.xlsx"
BANK_RATE_CSV   = "macro_Bank_Rate.csv"   # промежуточный CSV
OUT_PATH        = "data_macro_daily.csv"

# ===============================
# 0. КОНВЕРТАЦИЯ XLSX → CSV
# ===============================
print("=== Конвертация Bank Rate XLSX → CSV ===")

bank_xlsx = pd.read_excel(BANK_RATE_XLSX, header=0)

print("Колонки в xlsx:", bank_xlsx.columns.tolist())
print("Первые строки:")
print(bank_xlsx.head(5))
print("Последние строки:")
print(bank_xlsx.tail(3))

bank_xlsx.to_csv(BANK_RATE_CSV, index=False)
print(f"Сохранено: {BANK_RATE_CSV}\n")

# ===============================
# 1. ЗАГРУЗКА YIELD CURVE (спред 10y−2y)
# ===============================
yields = pd.read_csv(YIELD_PATH, index_col=0)
yields.index = pd.to_datetime(yields.index)
yields.index.name = "Date"

print("Колонки yield curve (первые 25):", yields.columns.tolist()[:25])

col_2y  = "2.0"
col_10y = "10.0"

spread_df = pd.DataFrame({
    "Spread_10y_2y": yields[col_10y].values - yields[col_2y].values
}, index=yields.index)

print(f"Yield curve: {len(spread_df)} дней, "
      f"{spread_df.index.min().date()} — {spread_df.index.max().date()}\n")

# ===============================
# 2. ЗАГРУЗКА ИНФЛЯЦИИ (месячные → дневные)
#    Формат строк: 1989 JAN,"4.9"
# ===============================
months_en = {
    "JAN":1,"FEB":2,"MAR":3,"APR":4,"MAY":5,"JUN":6,
    "JUL":7,"AUG":8,"SEP":9,"OCT":10,"NOV":11,"DEC":12
}

inf_rows = []
with open(INFLATION_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip().replace('"', '')
        if not line:
            continue
        parts = line.split(",")
        if len(parts) < 2:
            continue
        date_str = parts[0].strip()
        val_str  = parts[1].strip()
        try:
            year_str, mon_str = date_str.split()
            year  = int(year_str)
            month = months_en[mon_str.upper()]
            value = float(val_str)
            inf_rows.append((year, month, value))
        except Exception:
            continue

inf_monthly = pd.DataFrame(inf_rows, columns=["Year","Month","CPI"])
inf_monthly["Date"] = pd.to_datetime(
    inf_monthly[["Year","Month"]].assign(Day=1)
)
inf_monthly = inf_monthly.set_index("Date")[["CPI"]].sort_index()

print(f"Инфляция (месячная): {len(inf_monthly)} строк, "
      f"{inf_monthly.index.min().date()} — {inf_monthly.index.max().date()}")

# Растягиваем на каждый день через forward fill
all_days_inf = pd.date_range(
    inf_monthly.index.min(),
    inf_monthly.index.max() + pd.offsets.MonthEnd(0),
    freq="D"
)
inf_daily = inf_monthly.reindex(all_days_inf).ffill()
inf_daily.index.name = "Date"

print(f"Инфляция (дневная):  {len(inf_daily)} строк\n")

# ===============================
# 3. ЗАГРУЗКА BANK RATE из CSV
# ===============================
bank_raw = pd.read_csv(BANK_RATE_CSV, header=0)
bank_raw.columns = [c.strip() for c in bank_raw.columns]

print("Bank Rate колонки после конвертации:", bank_raw.columns.tolist())
print(bank_raw.head(3))
print()

# Определяем колонки автоматически
date_col = [c for c in bank_raw.columns
            if "date" in c.lower() or "Date" in c][0]
rate_col = [c for c in bank_raw.columns
            if "rate" in c.lower() or "Rate" in c][0]

bank_raw = bank_raw[[date_col, rate_col]].copy()
bank_raw.columns = ["Date_str", "Rate_str"]

# Чистим Rate: убираем пробелы, заменяем запятую на точку
bank_raw["Rate_str"] = (
    bank_raw["Rate_str"]
    .astype(str)
    .str.strip()
    .str.replace(",", ".", regex=False)
    .str.replace(" ", "", regex=False)
)

bank_raw["BankRate"] = pd.to_numeric(bank_raw["Rate_str"], errors="coerce")

# -------------------------------------------------------
# Парсим дату — пробуем несколько форматов
# "18 Dec 25", "18 Dec 2025", datetime уже распознанный
# -------------------------------------------------------
def parse_boe_date(s):
    s = str(s).strip()
    for fmt in ["%d %b %Y", "%d %b %y", "%Y-%m-%d", "%d/%m/%Y"]:
        try:
            dt = pd.to_datetime(s, format=fmt)
            # двузначный год: 25 → 2025, не 1925
            if dt.year < 1970:
                dt = dt.replace(year=dt.year + 100)
            return dt
        except Exception:
            continue
    # последняя попытка — pandas сам угадает
    try:
        return pd.to_datetime(s, dayfirst=True)
    except Exception:
        return pd.NaT

bank_raw["Date"] = bank_raw["Date_str"].apply(parse_boe_date)
bank_raw = bank_raw.dropna(subset=["Date", "BankRate"])
bank_raw = bank_raw.sort_values("Date").reset_index(drop=True)

print(f"Bank Rate: {len(bank_raw)} изменений")
print(f"Период: {bank_raw['Date'].min().date()} — "
      f"{bank_raw['Date'].max().date()}")
print(bank_raw.head(3))
print(bank_raw.tail(3))
print()

# Растягиваем на каждый день через forward fill
bank_indexed = bank_raw.set_index("Date")[["BankRate"]]

all_days_br = pd.date_range(
    bank_indexed.index.min(),
    bank_indexed.index.max(),
    freq="D"
)
bank_daily = bank_indexed.reindex(all_days_br).ffill()
bank_daily.index.name = "Date"

print(f"Bank Rate (дневной): {len(bank_daily)} строк\n")

# ===============================
# 4. ОБЪЕДИНЯЕМ ВСЁ

merged = spread_df.join(inf_daily,  how="left") \
                  .join(bank_daily, how="left")

merged = merged[
    (merged.index >= "1989-01-01") &
    (merged.index <= "2025-12-31")
]

# Добавь эту строку — заполняет 13 NaN в конце BankRate
merged = merged.ffill()

merged = merged.dropna(how="all")

print("=== Итоговый файл ===")
print(f"Строк:   {len(merged)}")
print(f"Период:  {merged.index.min().date()} — {merged.index.max().date()}")
print(f"Колонки: {merged.columns.tolist()}")
print(f"\nNaN по колонкам:\n{merged.isna().sum()}")
print(f"\nПервые строки:\n{merged.head()}")
print(f"\nПоследние строки:\n{merged.tail()}")

# ===============================
# 5. СОХРАНЯЕМ
# ===============================
merged.index.name = "Date"
merged.to_csv(OUT_PATH)
print(f"\nСохранено: {OUT_PATH}")