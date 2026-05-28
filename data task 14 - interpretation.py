import numpy as np
import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# ===============================
# PATHS
# ===============================
PCA_MODEL_PATH  = "artifacts_pca/pca_model.pkl"
AE_FACTORS_PATH = "artifacts_ae/ae_factors_train.csv"
TRAIN_PATH      = "UK_yield_curve_daily_processed_train.csv"
MACRO_PATH      = "data_macro_daily.csv"

OUT_DIR = "artifacts_interpretability"
os.makedirs(OUT_DIR, exist_ok=True)

# ===============================
# LOAD DATA
# ===============================
train = pd.read_csv(TRAIN_PATH)
maturities = train.columns[1:]
tenors = np.arange(len(maturities))

# ===============================
# LOAD PCA
# ===============================
pca = joblib.load(PCA_MODEL_PATH)
components = pca.components_
print("PCA components shape:", components.shape)

# ===============================
# PLOT FUNCTION (PCA)
# ===============================
def plot_pca_component(component, title, filename):
    plt.figure(figsize=(10, 5))
    plt.plot(tenors, component, marker="o")
    plt.axhline(0, color="black", linewidth=1)
    plt.xticks(tenors, maturities, rotation=45)
    plt.title(title)
    plt.grid()
    plt.tight_layout()
    path = os.path.join(OUT_DIR, filename)
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.show()
    plt.close()

# ===============================
# 1. PCA LEVEL / SLOPE / CURVATURE
# ===============================
plot_pca_component(components[0], "PCA Component 1 (Level)",     "pca_pc1_level.png")
plot_pca_component(components[1], "PCA Component 2 (Slope)",     "pca_pc2_slope.png")
plot_pca_component(components[2], "PCA Component 3 (Curvature)", "pca_pc3_curvature.png")

# ===============================
# 2. NORMALIZED SHAPES
# ===============================
norm_components = components / np.max(np.abs(components), axis=1, keepdims=True)
for i, name in enumerate(["level_norm", "slope_norm", "curvature_norm"]):
    plot_pca_component(norm_components[i], f"PCA {name}", f"pca_{name}.png")

# ===============================
# 3. EXPLAINED VARIANCE
# ===============================
plt.figure()
plt.bar(range(1, len(pca.explained_variance_ratio_) + 1),
        pca.explained_variance_ratio_)
plt.title("PCA Explained Variance Ratio")
plt.xlabel("Component")
plt.ylabel("Variance Share")
plt.tight_layout()
path = os.path.join(OUT_DIR, "pca_variance.png")
plt.savefig(path, dpi=150)
print(f"Saved: {path}")
plt.show()
plt.close()

# ===============================
# 4. AE FACTORS — загрузка и ранжирование
# ===============================
ae_factors = pd.read_csv(AE_FACTORS_PATH).values
n_latents = ae_factors.shape[1]
print(f"\nAE factors shape: {ae_factors.shape}")
print(f"Всего латентов: {n_latents}")

# Расчёт дисперсии каждого латента
variances = np.var(ae_factors, axis=0)

# Ранжирование от большей дисперсии к меньшей
ranking = np.argsort(variances)[::-1]  # индексы исходных столбцов в порядке убывания дисперсии

# Подсчёт мёртвых нейронов
dead_threshold = 1e-8
dead_count = int(np.sum(variances < dead_threshold))
active_count = n_latents - dead_count
print(f"Активных латентов: {active_count}")
print(f"Мёртвых латентов (дисперсия < {dead_threshold}): {dead_count}")

# ===============================
# 5. ТАБЛИЦА РАНЖИРОВАНИЯ ЛАТЕНТОВ
# ===============================
total_variance = np.sum(variances)

ranking_table = []
for rank, original_idx in enumerate(ranking, start=1):
    var_i = variances[original_idx]
    status = "DEAD" if var_i < dead_threshold else "active"
    ranking_table.append({
        "Rank":                rank,
        "Original_Latent_Idx": int(original_idx),
        "Variance":            round(float(var_i), 8),
        "Std":                 round(float(np.std(ae_factors[:, original_idx])), 6),
        "Mean":                round(float(np.mean(ae_factors[:, original_idx])), 6),
        "Min":                 round(float(np.min(ae_factors[:, original_idx])), 6),
        "Max":                 round(float(np.max(ae_factors[:, original_idx])), 6),
        "Range":               round(float(np.ptp(ae_factors[:, original_idx])), 6),
        "Variance_Share":      round(float(var_i / total_variance), 6) if total_variance > 0 else 0.0,
        "Status":              status,
    })

ranking_df = pd.DataFrame(ranking_table)
ranking_df.to_csv(os.path.join(OUT_DIR, "ae_latent_ranking.csv"), index=False)

print("\n=== AE Latent Ranking (по дисперсии, от больших к меньшим) ===")
print(ranking_df.to_string(index=False))

# ===============================
# 6. ГРАФИК-СВОДКА: ДИСПЕРСИИ ВСЕХ ЛАТЕНТОВ
# ===============================
plt.figure(figsize=(10, 5))
colors = ['#1E2761' if v >= dead_threshold else '#D14953' for v in variances[ranking]]
plt.bar(range(1, n_latents + 1), variances[ranking], color=colors)
plt.xlabel("Rank")
plt.ylabel("Variance")
plt.title("AE Latent Factors — variance by rank")
plt.xticks(range(1, n_latents + 1))
plt.axhline(dead_threshold, color="gray", linestyle="--", linewidth=0.8, label=f"Dead threshold = {dead_threshold}")
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
path = os.path.join(OUT_DIR, "ae_variance_ranking.png")
plt.savefig(path, dpi=150)
print(f"Saved: {path}")
plt.show()
plt.close()

# ===============================
# 7. ГРАФИКИ ТОП-3 ЛАТЕНТОВ (для основного текста)
# ===============================
def plot_ae_factor(values, title, filename):
    plt.figure(figsize=(10, 4))
    plt.plot(values)
    plt.title(title)
    plt.grid()
    plt.tight_layout()
    path = os.path.join(OUT_DIR, filename)
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.show()
    plt.close()

for rank in range(1, 4):
    original_idx = ranking[rank - 1]
    var_i = variances[original_idx]
    title = f"AE Top Factor {rank} (latent #{original_idx}, var = {var_i:.4f})"
    filename = f"ae_top{rank}.png"
    plot_ae_factor(ae_factors[:, original_idx], title, filename)

# ===============================
# 8. ВСЕ 16 ГРАФИКОВ (для приложения) — нумерация по рангу
# ===============================
for rank in range(1, n_latents + 1):
    original_idx = ranking[rank - 1]
    var_i = variances[original_idx]
    status = "DEAD" if var_i < dead_threshold else "active"
    title = f"AE Factor (rank {rank}) — latent #{original_idx}, var = {var_i:.6f} [{status}]"
    filename = f"ae_factor_rank{rank:02d}.png"
    plot_ae_factor(ae_factors[:, original_idx], title, filename)

# ===============================
# 9. SMOOTHNESS — только для PCA
# ===============================
def smoothness(x):
    return np.mean(np.abs(np.diff(x)))

pca_smooth = [smoothness(components[i]) for i in range(3)]
pca_smooth_df = pd.DataFrame({
    "Component":  ["Level (PC1)", "Slope (PC2)", "Curvature (PC3)"],
    "Smoothness": pca_smooth
})
pca_smooth_df.to_csv(os.path.join(OUT_DIR, "pca_smoothness.csv"), index=False)
print("\n=== PCA Smoothness (меньше = глаже) ===")
print(pca_smooth_df)

# ===============================
# 10. AE Top-3 — характеристики (заменяет старую "стабильность")
# ===============================
ae_top3_stability = []
for rank in range(1, 4):
    original_idx = ranking[rank - 1]
    ae_top3_stability.append({
        "Rank":        rank,
        "Latent_Idx":  int(original_idx),
        "Mean":   round(float(np.mean(ae_factors[:, original_idx])), 6),
        "Std":    round(float(np.std(ae_factors[:, original_idx])),  6),
        "Min":    round(float(np.min(ae_factors[:, original_idx])),  6),
        "Max":    round(float(np.max(ae_factors[:, original_idx])),  6),
        "Range":  round(float(np.ptp(ae_factors[:, original_idx])),  6),
    })
ae_top3_df = pd.DataFrame(ae_top3_stability)
ae_top3_df.to_csv(os.path.join(OUT_DIR, "ae_top3_stability.csv"), index=False)
print("\n=== AE Top-3 Factor Stability ===")
print(ae_top3_df)

# ===============================
# 11. PCA-ФАКТОРЫ: вычисляем на train
# ===============================
train_dates = pd.to_datetime(train.iloc[:, 0])

Y_train   = train[maturities].values
scaler    = StandardScaler()
Y_train_s = scaler.fit_transform(Y_train)

Z_train = pca.transform(Y_train_s)[:, :3]
print(f"\nZ_train shape (первые 3 компоненты): {Z_train.shape}")

pca_factors_df = pd.DataFrame(
    Z_train,
    columns=["Level", "Slope", "Curvature"]
)
pca_factors_df["Date"] = train_dates.values
pca_factors_df = pca_factors_df.set_index("Date")
pca_factors_df.index = pd.to_datetime(pca_factors_df.index)

# ===============================
# 12. ЗАГРУЖАЕМ МАКРО и МЁРДЖИМ
# ===============================
macro = pd.read_csv(MACRO_PATH, parse_dates=["Date"], index_col="Date")
print(f"\nМакро: {len(macro)} строк, "
      f"{macro.index.min().date()} — {macro.index.max().date()}")

combined = pca_factors_df.join(macro, how="inner").dropna()
print(f"После merge: {len(combined)} строк")

# ===============================
# 13. КОРРЕЛЯЦИЯ PCA С МАКРО
# ===============================
corr_cols   = ["Level", "Slope", "Curvature",
               "Spread_10y_2y", "CPI", "BankRate"]
corr_matrix = combined[corr_cols].corr()

print("\n=== Корреляция PCA-факторов с макропоказателями ===")
print(corr_matrix.round(4))
corr_matrix.to_csv(os.path.join(OUT_DIR, "pca_macro_correlation.csv"))

plt.figure(figsize=(9, 7))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, vmin=-1, vmax=1, square=True, linewidths=0.5)
plt.title("Correlation: PCA Factors vs Macro Indicators", fontsize=13)
plt.tight_layout()
path = os.path.join(OUT_DIR, "pca_macro_heatmap.png")
plt.savefig(path, dpi=150)
print(f"Saved: {path}")
plt.show()
plt.close()

# ===============================
# 13b. КОРРЕЛЯЦИЯ PCA С МАКРО НА ПЕРВЫХ РАЗНОСТЯХ
# (страховка от spurious correlation двух персистентных рядов)
# ===============================
combined_diff = combined[corr_cols].diff().dropna()
corr_matrix_diff = combined_diff.corr()

print("\n=== Корреляция PCA-факторов с макропоказателями (первые разности) ===")
print(corr_matrix_diff.round(4))
corr_matrix_diff.to_csv(os.path.join(OUT_DIR, "pca_macro_correlation_diff.csv"))

plt.figure(figsize=(9, 7))
sns.heatmap(corr_matrix_diff, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, vmin=-1, vmax=1, square=True, linewidths=0.5)
plt.title("Correlation: ΔPCA Factors vs ΔMacro Indicators (first differences)", fontsize=13)
plt.tight_layout()
path = os.path.join(OUT_DIR, "pca_macro_heatmap_diff.png")
plt.savefig(path, dpi=150)
print(f"Saved: {path}")
plt.show()
plt.close()

# Сравнение уровни vs разности — компактная сводка
print("\n=== Сравнение корреляций: уровни vs первые разности ===")
pairs = [("Level", "BankRate"),
         ("Slope", "Spread_10y_2y"),
         ("Curvature", "CPI"),
         ("Level", "CPI"),
         ("Slope", "BankRate"),
         ("Curvature", "BankRate")]
cmp_rows = []
for f, m in pairs:
    r_levels = corr_matrix.loc[f, m]
    r_diffs  = corr_matrix_diff.loc[f, m]
    cmp_rows.append({
        "Factor": f,
        "Macro":  m,
        "Corr_levels":     round(float(r_levels), 4),
        "Corr_diffs":      round(float(r_diffs),  4),
        "Delta_abs":       round(float(abs(r_levels) - abs(r_diffs)), 4),
    })
cmp_df = pd.DataFrame(cmp_rows)
cmp_df.to_csv(os.path.join(OUT_DIR, "pca_macro_levels_vs_diffs.csv"), index=False)
print(cmp_df.to_string(index=False))

# ===============================
# 14. КОРРЕЛЯЦИЯ AE (TOP-3) С МАКРО
# ===============================
n_rows = min(len(train_dates), len(ae_factors))

# берём топ-3 латента по дисперсии и нумеруем их с 1
top3_indices = ranking[:3]
top3_columns = [f"AE_TopFactor_{r+1}" for r in range(3)]

ae_top3_data = ae_factors[:n_rows, top3_indices]
ae_df = pd.DataFrame(
    ae_top3_data,
    columns=top3_columns,
    index=pd.to_datetime(train_dates.values[:n_rows])
)
ae_df.index.name = "Date"

combined_ae = ae_df.join(macro, how="inner").dropna()
print(f"\nAE Top-3 + macro: {len(combined_ae)} строк")

ae_corr_cols = top3_columns + ["Spread_10y_2y", "CPI", "BankRate"]
ae_corr_matrix = combined_ae[ae_corr_cols].corr()

print("\n=== Корреляция AE Top-3 факторов с макропоказателями ===")
print(ae_corr_matrix.round(4))
ae_corr_matrix.to_csv(os.path.join(OUT_DIR, "ae_macro_correlation.csv"))

plt.figure(figsize=(9, 7))
sns.heatmap(ae_corr_matrix, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, vmin=-1, vmax=1, square=True, linewidths=0.5)
plt.title("Correlation: AE Top-3 Factors vs Macro Indicators", fontsize=13)
plt.tight_layout()
path = os.path.join(OUT_DIR, "ae_macro_heatmap.png")
plt.savefig(path, dpi=150)
print(f"Saved: {path}")
plt.show()
plt.close()

# ===============================
# 14b. КОРРЕЛЯЦИЯ AE (TOP-3) С МАКРО НА ПЕРВЫХ РАЗНОСТЯХ
# ===============================
combined_ae_diff = combined_ae[ae_corr_cols].diff().dropna()
ae_corr_matrix_diff = combined_ae_diff.corr()

print("\n=== Корреляция AE Top-3 факторов с макропоказателями (первые разности) ===")
print(ae_corr_matrix_diff.round(4))
ae_corr_matrix_diff.to_csv(os.path.join(OUT_DIR, "ae_macro_correlation_diff.csv"))

plt.figure(figsize=(9, 7))
sns.heatmap(ae_corr_matrix_diff, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, vmin=-1, vmax=1, square=True, linewidths=0.5)
plt.title("Correlation: ΔAE Top-3 Factors vs ΔMacro Indicators (first differences)", fontsize=13)
plt.tight_layout()
path = os.path.join(OUT_DIR, "ae_macro_heatmap_diff.png")
plt.savefig(path, dpi=150)
print(f"Saved: {path}")
plt.show()
plt.close()

# ===============================
# 15. СВОДНАЯ ТАБЛИЦА PCA vs AE TOP-3
# ===============================
rows = []
for factor in ["Level", "Slope", "Curvature"]:
    for mc in ["Spread_10y_2y", "CPI", "BankRate"]:
        r = combined[[factor, mc]].corr().iloc[0, 1]
        rows.append({
            "Method": "PCA",
            "Factor": factor,
            "Macro":  mc,
            "Corr":   round(r, 4)
        })

for rank, col in enumerate(top3_columns, start=1):
    for mc in ["Spread_10y_2y", "CPI", "BankRate"]:
        r = combined_ae[[col, mc]].corr().iloc[0, 1]
        rows.append({
            "Method": "AE",
            "Factor": f"TopFactor_{rank} (latent #{top3_indices[rank-1]})",
            "Macro":  mc,
            "Corr":   round(r, 4)
        })

summary_df = pd.DataFrame(rows)
summary_df.to_csv(
    os.path.join(OUT_DIR, "interpretability_summary.csv"), index=False
)

print("\n=== Сводная таблица корреляций PCA vs AE Top-3 ===")
print(summary_df.to_string(index=False))
print(f"\nВсе результаты сохранены в: {OUT_DIR}")