import pandas as pd

excel_file = "Yield curves up to 25 daily.xlsx"
xls = pd.ExcelFile(excel_file)
sheet_name = xls.sheet_names[0]

df = pd.read_excel(excel_file, sheet_name=sheet_name, header=1)  # header=1 если первая строка — это заголовки столбцов

csv_file = "UK_yield_curve daily.csv"
df.to_csv(csv_file, index=False)

print(f"Файл сохранен как {csv_file}")