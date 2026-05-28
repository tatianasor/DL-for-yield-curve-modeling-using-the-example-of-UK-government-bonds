import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.preprocessing import StandardScaler

# ===============================
# PATHS
# ===============================
PCA_MODEL_PATH = "artifacts_pca/pca_model.pkl"
YIELD_PATH     = "UK_yield_curve_daily_processed.csv"
OUT_DIR        = "artifacts_step15"
os.makedirs(OUT_DIR, exist_ok=True)

# ===============================
# 1. ЗАГРУЗКА ДАННЫХ
# ===============================
pca    = joblib.load(PCA_MODEL_PATH)
yields = pd.read_csv(YIELD_PATH, index_col=0)
yields.index = pd.to_datetime(yields.index)
yields = yields.dropna(how="all")

maturities = yields.columns.astype(float)  # [0.5, 1.0, ..., 25.0]

print(f"Yield curve: {yields.shape}")
print(f"Период: {yields.index.min().date()} — {yields.index.max().date()}")
print(f"Тенора: {maturities.tolist()[:5]} ... {maturities.tolist()[-5:]}")

# ===============================
# 2. ТЕКУЩАЯ КРИВАЯ
#    Берём последнюю доступную дату тестовой выборки
#    Тест = последние 15% данных
# ===============================
n_total  = len(yields)
n_test   = int(n_total * 0.15)
test_data = yields.iloc[-n_test:]

# "Текущая" дата — последний день теста
current_date  = test_data.index[-1]
current_curve = test_data.iloc[-1].values  # форма (50,)

print(f"\nТекущая дата: {current_date.date()}")
print(f"Текущая кривая (первые 5 тенора): "
      f"{current_curve[:5].round(4)}")

# ===============================
# 3. PCA-ФАКТОРЫ НА ТЕСТОВЫХ ДАННЫХ
#    Считаем шоки = дневные изменения факторов
# ===============================
scaler    = StandardScaler()

# Обучаем scaler только на train (первые 70%)
n_train   = int(n_total * 0.70)
train_data = yields.iloc[:n_train]

scaler.fit(train_data.values)

# Трансформируем тест
test_scaled   = scaler.transform(test_data.values)
test_factors  = pca.transform(test_scaled)[:, :3]  # (N_test, 3)

# Дневные изменения факторов = шоки
factor_shocks = np.diff(test_factors, axis=0)  # (N_test-1, 3)

print(f"\nШоков (дней): {len(factor_shocks)}")
print(f"Статистика шоков (Level, Slope, Curvature):")
shock_stats = pd.DataFrame(factor_shocks,
                            columns=["Level_shock",
                                     "Slope_shock",
                                     "Curvature_shock"])
print(shock_stats.describe().round(6))

# ===============================
# 4. ФУНКЦИЯ ЦЕНООБРАЗОВАНИЯ ОБЛИГАЦИИ
#    Цена = сумма дисконтированных купонов + номинал
#    Используем yield для данного срока (линейная интерполяция)
# ===============================
def bond_price(face, coupon_rate, maturity_years, yield_curve_values,
               yield_maturities):
    """
    face            — номинал (£)
    coupon_rate     — годовой купон (доли, напр. 0.04)
    maturity_years  — срок до погашения
    yield_curve     — массив доходностей по тенорам
    yield_maturities— массив тенóров (лет)
    """
    coupon   = face * coupon_rate
    price    = 0.0

    for t in range(1, int(maturity_years * 2) + 1):
        # полугодовые купоны
        t_years = t / 2.0
        if t_years > maturity_years:
            break
        # интерполируем доходность для срока t_years
        y = np.interp(t_years, yield_maturities, yield_curve_values) / 100
        cf = coupon / 2  # полугодовой купон
        if t == int(maturity_years * 2):
            cf += face  # последний платёж + номинал
        price += cf / (1 + y / 2) ** t

    return price

# ===============================
# 5. ПОРТФЕЛЬ
# ===============================
portfolio = [
    {"name": "Short  (2y)",  "face": 1_000_000, "coupon": 0.04,
     "maturity": 2.0},
    {"name": "Medium (10y)", "face": 1_000_000, "coupon": 0.04,
     "maturity": 10.0},
    {"name": "Long   (25y)", "face": 1_000_000, "coupon": 0.04,
     "maturity": 25.0},
]

# Текущая стоимость портфеля
current_prices = []
for bond in portfolio:
    p = bond_price(bond["face"], bond["coupon"], bond["maturity"],
                   current_curve, maturities)
    current_prices.append(p)
    print(f"{bond['name']}: £{p:,.2f}")

current_portfolio_value = sum(current_prices)
print(f"\nТекущая стоимость портфеля: £{current_portfolio_value:,.2f}")

# ===============================
# 6. ИСТОРИЧЕСКАЯ СИМУЛЯЦИЯ
#    Для каждого исторического шока:
#    1. Применяем шок к текущим факторам
#    2. Восстанавливаем кривую через inverse_transform
#    3. Считаем новую стоимость портфеля
#    4. P&L = новая стоимость - текущая
# ===============================

# Текущие факторы (на последний день теста)
current_scaled  = scaler.transform(current_curve.reshape(1, -1))
current_factors = pca.transform(current_scaled)[0, :3]  # (3,)

print(f"\nТекущие PCA-факторы (Level, Slope, Curvature): "
      f"{current_factors.round(4)}")

pnl_list       = []
shocked_curves = []

for shock in factor_shocks:
    # Шокированные факторы
    shocked_factors = current_factors + shock  # (3,)

    # Восстанавливаем кривую:
    # нужно передать полный вектор факторов (все n_components)
    full_factors = pca.transform(current_scaled)[0].copy()
    full_factors[:3] = shocked_factors

    # Обратное преобразование: факторы → scaled yields → yields
    shocked_scaled = full_factors @ pca.components_ + pca.mean_
    shocked_curve  = scaler.inverse_transform(
        shocked_scaled.reshape(1, -1)
    )[0]

    shocked_curves.append(shocked_curve)

    # Новая стоимость портфеля
    new_prices = []
    for bond in portfolio:
        p = bond_price(bond["face"], bond["coupon"], bond["maturity"],
                       shocked_curve, maturities)
        new_prices.append(p)

    new_value = sum(new_prices)
    pnl_list.append(new_value - current_portfolio_value)

pnl = np.array(pnl_list)

print(f"\nСимуляций: {len(pnl)}")
print(f"P&L статистика:")
pnl_stats = pd.Series(pnl)
print(pnl_stats.describe().apply(lambda x: f"£{x:,.2f}"))

# ===============================
# 7. VAR РАСЧЁТ
# ===============================
var_95  = np.percentile(pnl, 5)   # 5-й перцентиль = VaR 95%
var_99  = np.percentile(pnl, 1)   # 1-й перцентиль = VaR 99%
cvar_95 = pnl[pnl <= var_95].mean()  # CVaR (Expected Shortfall)
cvar_99 = pnl[pnl <= var_99].mean()

print(f"\n{'='*45}")
print(f"  РЕЗУЛЬТАТЫ VaR (историческая симуляция)")
print(f"{'='*45}")
print(f"  Текущая стоимость портфеля: "
      f"£{current_portfolio_value:>15,.2f}")
print(f"  Кол-во сценариев:           {len(pnl):>15,}")
print(f"{'='*45}")
print(f"  VaR(95%,  1 день):          £{var_95:>15,.2f}")
print(f"  VaR(99%,  1 день):          £{var_99:>15,.2f}")
print(f"  CVaR(95%, 1 день):          £{cvar_95:>15,.2f}")
print(f"  CVaR(99%, 1 день):          £{cvar_99:>15,.2f}")
print(f"{'='*45}")
print(f"  VaR(95%) как % портфеля:   "
      f"{abs(var_95)/current_portfolio_value*100:>14.3f}%")
print(f"  VaR(99%) как % портфеля:   "
      f"{abs(var_99)/current_portfolio_value*100:>14.3f}%")

# ===============================
# 8. ТАБЛИЦА ДЛЯ ДИПЛОМА
# ===============================
results_table = pd.DataFrame({
    "Метрика": [
        "Стоимость портфеля (£)",
        "Кол-во сценариев",
        "VaR(95%), 1 день (£)",
        "VaR(99%), 1 день (£)",
        "CVaR(95%), 1 день (£)",
        "CVaR(99%), 1 день (£)",
        "VaR(95%) как % портфеля",
        "VaR(99%) как % портфеля",
    ],
    "Значение": [
        f"£{current_portfolio_value:,.2f}",
        f"{len(pnl):,}",
        f"£{var_95:,.2f}",
        f"£{var_99:,.2f}",
        f"£{cvar_95:,.2f}",
        f"£{cvar_99:,.2f}",
        f"{abs(var_95)/current_portfolio_value*100:.3f}%",
        f"{abs(var_99)/current_portfolio_value*100:.3f}%",
    ]
})
results_table.to_csv(os.path.join(OUT_DIR, "var_results.csv"), index=False)
print(f"\nТаблица сохранена: {OUT_DIR}/var_results.csv")

# ===============================
# 9. ГРАФИКИ
# ===============================

# --- 9.1 Распределение P&L ---
fig, ax = plt.subplots(figsize=(12, 6))

ax.hist(pnl, bins=80, color="steelblue", alpha=0.7,
        edgecolor="white", linewidth=0.3)
ax.axvline(var_95,  color="orange", linewidth=2,
           linestyle="--", label=f"VaR(95%)  = £{var_95:,.0f}")
ax.axvline(var_99,  color="red",    linewidth=2,
           linestyle="--", label=f"VaR(99%)  = £{var_99:,.0f}")
ax.axvline(cvar_95, color="orange", linewidth=2,
           linestyle=":",  label=f"CVaR(95%) = £{cvar_95:,.0f}",
           alpha=0.8)
ax.axvline(cvar_99, color="red",    linewidth=2,
           linestyle=":",  label=f"CVaR(99%) = £{cvar_99:,.0f}",
           alpha=0.8)
ax.axvline(0, color="black", linewidth=1, alpha=0.5)

ax.set_xlabel("P&L (£)", fontsize=12)
ax.set_ylabel("Частота", fontsize=12)
ax.set_title(
    f"Распределение P&L портфеля UK Gilts\n"
    f"Историческая симуляция по PCA-факторам | "
    f"Текущая дата: {current_date.date()}",
    fontsize=13
)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
path = os.path.join(OUT_DIR, "pnl_distribution.png")
plt.savefig(path, dpi=150)
print(f"Saved: {path}")
plt.show()
plt.close()

# --- 9.2 Шокированные кривые (50 сценариев) ---
fig, ax = plt.subplots(figsize=(12, 6))

# Рисуем 50 случайных сценариев
idx_sample = np.random.choice(len(shocked_curves), 50, replace=False)
for i in idx_sample:
    ax.plot(maturities, shocked_curves[i],
            color="steelblue", alpha=0.15, linewidth=0.8)

ax.plot(maturities, current_curve,
        color="red", linewidth=2.5, label=f"Текущая ({current_date.date()})")

ax.set_xlabel("Срок до погашения (лет)", fontsize=12)
ax.set_ylabel("Доходность (%)", fontsize=12)
ax.set_title(
    "Симулированные кривые доходности UK Gilts\n"
    "50 исторических сценариев PCA-шоков",
    fontsize=13
)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
path = os.path.join(OUT_DIR, "shocked_curves.png")
plt.savefig(path, dpi=150)
print(f"Saved: {path}")
plt.show()
plt.close()

# --- 9.3 Вклад каждой облигации в VaR ---

bond_pnl = {bond["name"]: [] for bond in portfolio}

for shock in factor_shocks:
    full_factors       = pca.transform(current_scaled)[0].copy()
    full_factors[:3]   = current_factors + shock
    shocked_scaled_row = full_factors @ pca.components_ + pca.mean_
    shocked_curve_row  = scaler.inverse_transform(
        shocked_scaled_row.reshape(1, -1)
    )[0]

    for bond, cp in zip(portfolio, current_prices):
        p = bond_price(bond["face"], bond["coupon"], bond["maturity"],
                       shocked_curve_row, maturities)
        bond_pnl[bond["name"]].append(p - cp)

fig, axes = plt.subplots(1, 3, figsize=(15, 6))  # увеличили высоту

for ax, bond in zip(axes, portfolio):
    bp = np.array(bond_pnl[bond["name"]])
    v95 = np.percentile(bp, 5)
    ax.hist(bp, bins=60, color="steelblue", alpha=0.7,
            edgecolor="white", linewidth=0.3)
    ax.axvline(v95, color="red", linewidth=2,
               linestyle="--", label=f"VaR(95%)=£{v95:,.0f}")
    ax.set_title(bond["name"], fontsize=12)
    ax.set_xlabel("P&L (£)", fontsize=10)
    ax.set_ylabel("Частота", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

# ИСПРАВЛЕНИЕ: title внутри figure через fig.text вместо suptitle
fig.suptitle(
    "P&L по отдельным облигациям портфеля",
    fontsize=13,
    y=0.98        # опускаем ниже — было y=1.02 (выше картинки)
)

plt.tight_layout(rect=[0, 0, 1, 0.95])  # оставляем место для title сверху

path = os.path.join(OUT_DIR, "bond_pnl_breakdown.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
print(f"Saved: {path}")
plt.show()
plt.close()

print(f"\nВсе результаты сохранены в: {OUT_DIR}/")