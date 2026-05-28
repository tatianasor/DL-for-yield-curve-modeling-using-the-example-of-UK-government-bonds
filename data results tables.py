import pandas as pd
import numpy as np

# -----------------------------
# 1. Load reconstructed curves
# -----------------------------
look_back = 18

test_orig = pd.read_csv("UK_yield_curve_processed_test.csv")
pca = pd.read_csv("reconstructed_curve_PCA_test.csv")
pca_ann = pd.read_csv("reconstructed_curve_PCA_ANN_test.csv")
pca_lstm = pd.read_csv("reconstructed_curve_PCA_LSTM_test.csv")
ae = pd.read_csv("reconstructed_curve_AE_test.csv")
ae_ann = pd.read_csv("reconstructed_curve_AE_ANN_test.csv")
ae_lstm = pd.read_csv("reconstructed_curve_AE_LSTM_test.csv")

# -----------------------------
# 2. Maturities
# -----------------------------
maturities = test_orig.columns[1:]
maturities_float = [float(m) for m in maturities]

# выбираем maturities с шагом 0.5
selected_maturities = [m for m in maturities_float if (m*2) % 1 == 0]

# -----------------------------
# 3. Selected test dates
# -----------------------------
test_dates_idx = [0, len(test_orig)//4, len(test_orig)//2, 3*len(test_orig)//4, len(test_orig)-1]

# -----------------------------
# 4. Excel writer
# -----------------------------
excel_writer = pd.ExcelWriter("yield_curves_comparison.xlsx", engine='xlsxwriter')

# -----------------------------
# 5. Create sheets
# -----------------------------
for idx in test_dates_idx:
    date_label = test_orig["Date"].iloc[idx]

    # формируем словарь с моделями
    models_dict = {
        "Original": test_orig.iloc[idx, 1:].values,
        "PCA": pca.iloc[idx, 1:].values,
        "PCA + ANN": pca_ann.iloc[idx, 1:].values,
        "AE": ae.iloc[idx, 1:].values,
        "AE + ANN": ae_ann.iloc[idx, 1:].values
    }

    # LSTM только если есть данные
    if idx >= look_back:
        lstm_idx = idx - look_back
        if lstm_idx < len(pca_lstm):
            models_dict["PCA + LSTM"] = pca_lstm.iloc[lstm_idx, 1:].values
        if lstm_idx < len(ae_lstm):
            models_dict["AE + LSTM"] = ae_lstm.iloc[lstm_idx, 1:].values

    # формируем таблицу по выбранным maturities
    excel_data = {}
    for model_name, values in models_dict.items():
        selected_values = [values[maturities_float.index(m)] for m in selected_maturities]
        excel_data[model_name] = selected_values

    df_excel = pd.DataFrame(excel_data, index=[str(int(m)) if m.is_integer() else m for m in selected_maturities])
    df_excel = df_excel.transpose()
    df_excel.index.name = "Model"

    # добавляем лист
    sheet_name = f"Date_{date_label}"
    df_excel.to_excel(excel_writer, sheet_name=sheet_name)

# -----------------------------
# 6. Save Excel
# -----------------------------
excel_writer.close()
print("Excel file 'yield_curves_comparison.xlsx' created with 5 sheets for 5 dates.")