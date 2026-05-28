import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load data
# -----------------------------
test_orig = pd.read_csv("UK_yield_curve_processed_test.csv")

pca = pd.read_csv("reconstructed_curve_PCA_test.csv")
pca_ann = pd.read_csv("reconstructed_curve_PCA_ANN_test.csv")
pca_lstm = pd.read_csv("reconstructed_curve_PCA_LSTM_test.csv")

ae = pd.read_csv("reconstructed_curve_AE_test.csv")
ae_ann = pd.read_csv("reconstructed_curve_AE_ANN_test.csv")
ae_lstm = pd.read_csv("reconstructed_curve_AE_LSTM_test.csv")

look_back = 18

# -----------------------------
# 2. Maturities
# -----------------------------
maturities = test_orig.columns[1:]
maturities_float = [float(m) for m in maturities]
xticks = [int(m) for m in maturities_float if float(m).is_integer()]

# -----------------------------
# 3. Dates
# -----------------------------
test_dates_idx = [0, len(test_orig)//4, len(test_orig)//2,
                  3*len(test_orig)//4, len(test_orig)-1]

# -----------------------------
# 4. Plot
# -----------------------------
for idx in test_dates_idx:

    date_label = test_orig["Date"].iloc[idx]
    plt.figure(figsize=(13, 6))

    # --- ORIGINAL ---
    plt.plot(maturities_float,
             test_orig.iloc[idx, 1:].values,
             color='limegreen',
             linewidth=2.3,
             marker='o',
             markersize=5,
             label="Original")

    # --- PCA ---
    plt.plot(maturities_float,
             pca.iloc[idx, 1:].values,
             color='#5DADE2',
             linewidth=1.6,
             marker='o',
             markersize=4,
             label="PCA")

    plt.plot(maturities_float,
             pca_ann.iloc[idx, 1:].values,
             color='#1B4F72',
             linewidth=1.6,
             marker='o',
             markersize=4,
             label="PCA + ANN")

    # --- AE ---
    plt.plot(maturities_float,
             ae.iloc[idx, 1:].values,
             color='#EC7063',
             linewidth=1.6,
             marker='o',
             markersize=4,
             label="AE")

    plt.plot(maturities_float,
             ae_ann.iloc[idx, 1:].values,
             color='#641E16',
             linewidth=1.6,
             marker='o',
             markersize=4,
             label="AE + ANN")

    # -----------------------------
    # LSTM (ТОЛЬКО если есть данные)
    # -----------------------------
    if idx >= look_back:
        lstm_idx = idx - look_back

        # PCA LSTM
        if lstm_idx < len(pca_lstm):
            plt.plot(maturities_float,
                     pca_lstm.iloc[lstm_idx, 1:].values,
                     color='#2E86C1',
                     linewidth=1.6,
                     marker='o',
                     markersize=4,
                     label="PCA + LSTM")

        # AE LSTM
        if lstm_idx < len(ae_lstm):
            plt.plot(maturities_float,
                     ae_lstm.iloc[lstm_idx, 1:].values,
                     color='#C0392B',
                     linewidth=1.6,
                     marker='o',
                     markersize=4,
                     label="AE + LSTM")

    # -----------------------------
    # Formatting
    # -----------------------------
    plt.xlabel("Maturity (years)", fontname="Times New Roman", fontsize=12)
    plt.ylabel("Yield (%)", fontname="Times New Roman", fontsize=12)
    plt.title(f"Yield Curve Comparison Across Models\nDate: {date_label}",
              fontname="Times New Roman", fontsize=14)

    plt.xticks(xticks, xticks, rotation=45)

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)

    plt.grid(True)
    plt.tight_layout(rect=[0, 0, 0.8, 1])

    plt.show()