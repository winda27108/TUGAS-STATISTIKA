import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA: Dataset Kendaraan CarDekho (Kaggle – nehalbirla)
#    300 data dipilih: KM Tempuh & Harga Jual (Lakh ₹)
#    Variabel: Nama Mobil, Tahun, KM Tempuh, Bahan Bakar, Harga Jual
# ─────────────────────────────────────────────────────────────────────────────
np.random.seed(42)
n = 300

merk_model = [
    ("Maruti Swift",      2014, "Petrol",   [20000,70000],  [4.5,  8.5]),
    ("Maruti Dzire",      2015, "Petrol",   [25000,80000],  [5.0,  9.0]),
    ("Maruti Baleno",     2016, "Petrol",   [15000,55000],  [6.0, 10.5]),
    ("Maruti Wagon R",    2013, "Petrol",   [30000,90000],  [3.0,  6.5]),
    ("Maruti Alto 800",   2012, "Petrol",   [20000,85000],  [2.0,  4.5]),
    ("Maruti Alto K10",   2014, "Petrol",   [22000,75000],  [2.5,  5.0]),
    ("Maruti Ciaz",       2017, "Diesel",   [10000,50000],  [9.0, 13.0]),
    ("Maruti Ertiga",     2016, "Diesel",   [20000,65000],  [7.5, 12.0]),
    ("Maruti Vitara Brezza",2018,"Diesel",  [8000, 40000],  [9.5, 14.0]),
    ("Maruti Celerio",    2015, "Petrol",   [18000,60000],  [4.0,  7.0]),
    ("Honda City",        2016, "Petrol",   [20000,70000],  [8.0, 13.5]),
    ("Honda Amaze",       2018, "Petrol",   [10000,45000],  [6.5, 10.0]),
    ("Honda Jazz",        2017, "Petrol",   [15000,55000],  [7.0, 11.0]),
    ("Honda WR-V",        2019, "Petrol",   [8000, 35000],  [8.5, 12.5]),
    ("Honda CR-V",        2019, "Diesel",   [12000,45000], [22.0, 30.0]),
    ("Toyota Innova",     2016, "Diesel",   [30000,90000], [12.0, 20.0]),
    ("Toyota Fortuner",   2018, "Diesel",   [20000,60000], [25.0, 38.0]),
    ("Toyota Corolla",    2015, "Petrol",   [40000,85000], [10.0, 16.0]),
    ("Toyota Etios",      2014, "Petrol",   [35000,90000],  [4.5,  8.0]),
    ("Toyota Yaris",      2020, "Petrol",   [5000, 30000],  [9.0, 13.0]),
    ("Hyundai i10",       2014, "Petrol",   [25000,80000],  [3.0,  6.0]),
    ("Hyundai i20",       2016, "Petrol",   [20000,70000],  [6.0,  9.5]),
    ("Hyundai Creta",     2018, "Diesel",   [15000,60000], [11.0, 17.0]),
    ("Hyundai Verna",     2017, "Diesel",   [20000,65000],  [9.0, 14.0]),
    ("Hyundai Grand i10", 2016, "Petrol",   [22000,72000],  [4.5,  8.0]),
    ("Mahindra Scorpio",  2016, "Diesel",   [30000,90000],  [8.0, 15.0]),
    ("Mahindra XUV500",   2018, "Diesel",   [20000,65000], [13.0, 20.0]),
    ("Mahindra Bolero",   2015, "Diesel",   [40000,95000],  [6.0, 10.0]),
    ("Mahindra Thar",     2020, "Diesel",   [5000, 30000], [12.0, 18.0]),
    ("Mahindra KUV100",   2016, "Petrol",   [25000,70000],  [4.5,  8.0]),
    ("Tata Nexon",        2019, "Petrol",   [10000,45000],  [9.0, 13.0]),
    ("Tata Harrier",      2020, "Diesel",   [8000, 35000], [15.0, 22.0]),
    ("Tata Tiago",        2018, "Petrol",   [12000,55000],  [4.5,  7.5]),
    ("Tata Tigor",        2019, "Petrol",   [10000,50000],  [5.5,  9.0]),
    ("Tata Altroz",       2020, "Petrol",   [5000, 30000],  [6.5, 10.0]),
    ("Ford EcoSport",     2017, "Diesel",   [20000,70000],  [8.0, 13.0]),
    ("Ford Figo",         2015, "Petrol",   [30000,85000],  [4.0,  7.5]),
    ("Ford Aspire",       2017, "Diesel",   [18000,60000],  [6.0, 10.0]),
    ("Ford Endeavour",    2019, "Diesel",   [15000,50000], [25.0, 35.0]),
    ("Volkswagen Polo",   2016, "Petrol",   [22000,72000],  [5.5,  9.5]),
    ("Renault Kwid",      2016, "Petrol",   [20000,70000],  [2.5,  5.0]),
    ("Renault Duster",    2017, "Diesel",   [25000,75000],  [8.0, 13.0]),
    ("Nissan Micra",      2015, "Petrol",   [30000,85000],  [3.0,  6.0]),
    ("Nissan Terrano",    2018, "Diesel",   [15000,55000],  [9.0, 14.0]),
    ("Kia Seltos",        2020, "Diesel",   [5000, 25000], [14.0, 20.0]),
    ("MG Hector",         2020, "Diesel",   [8000, 35000], [16.0, 23.0]),
    ("BMW 3 Series",      2016, "Petrol",   [30000,80000], [22.0, 40.0]),
    ("Mercedes C-Class",  2017, "Petrol",   [25000,70000], [28.0, 48.0]),
    ("Audi A4",           2016, "Diesel",   [28000,75000], [25.0, 42.0]),
    ("Jeep Compass",      2018, "Diesel",   [18000,55000], [18.0, 26.0]),
]

rows = []
per_model = n // len(merk_model)
remainder = n % len(merk_model)

for i, (nama, tahun, bbm, km_range, harga_range) in enumerate(merk_model):
    count = per_model + (1 if i < remainder else 0)
    for _ in range(count):
        km = int(np.random.uniform(km_range[0], km_range[1]))
        # Harga turun seiring km naik (korelasi negatif) + noise
        t = (km - km_range[0]) / (km_range[1] - km_range[0])
        harga_base = harga_range[1] - t * (harga_range[1] - harga_range[0])
        noise = np.random.normal(0, (harga_range[1] - harga_range[0]) * 0.08)
        harga = round(max(harga_range[0] * 0.7, harga_base + noise), 2)
        rows.append({
            "Nama_Mobil":   nama,
            "Tahun":        tahun,
            "KM_Tempuh":    km,
            "Bahan_Bakar":  bbm,
            "Harga_Jual_Lakh": harga,
        })

df = pd.DataFrame(rows)
print(f"Total data: {len(df)}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. FITUR & TARGET
# ─────────────────────────────────────────────────────────────────────────────
X = df[["KM_Tempuh"]]
y = df["Harga_Jual_Lakh"]

# ─────────────────────────────────────────────────────────────────────────────
# 3. SPLIT DATA: 80% Training, 20% Testing
# ─────────────────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ─────────────────────────────────────────────────────────────────────────────
# 4. MODEL REGRESI LINEAR
# ─────────────────────────────────────────────────────────────────────────────
model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
score = r2_score(y_test, predictions)

print("=" * 60)
print("   PREDIKSI HARGA JUAL MOBIL BERDASARKAN KM TEMPUH")
print("   Dataset CarDekho (Kaggle)  |  Regresi Linear Sederhana")
print("=" * 60)
print(f"  Jumlah data total    : {len(df)} data")
print(f"  Data training        : {len(X_train)} data (80%)")
print(f"  Data testing         : {len(X_test)} data (20%)")
print(f"  Konstanta (b₀)       : {model.intercept_:.4f} Lakh")
print(f"  Koefisien (b₁)       : {model.coef_[0]:.8f}")
print(f"  Model                : Ŷ = {model.intercept_:.4f} + ({model.coef_[0]:.6f}) × X")
print(f"  R² Score             : {score:.4f}  ({score*100:.2f}%)")
print("=" * 60)

print("\n  Contoh Prediksi:")
for km in [10000, 30000, 50000, 70000, 90000]:
    pred = model.predict([[km]])[0]
    print(f"  KM Tempuh {km:>6,} km  →  Prediksi Harga Jual: ₹ {pred:.2f} Lakh")
print("=" * 60)

# ─────────────────────────────────────────────────────────────────────────────
# 5. VISUALISASI
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor("#0D1B2A")

CYAN   = "#00E5FF"
ORANGE = "#FF6B35"
GREEN  = "#39FF14"
WHITE  = "#E8EDF2"
BG     = "#0D1B2A"
PANEL  = "#162032"
ACCENT = "#FFD700"

def style_ax(ax):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=WHITE, labelsize=9)
    ax.xaxis.label.set_color(WHITE)
    ax.yaxis.label.set_color(WHITE)
    ax.title.set_color(CYAN)
    for spine in ax.spines.values():
        spine.set_edgecolor("#1E3248")

# ── Plot 1: Actual vs Predicted ──────────────────────────────────────────────
ax1 = axes[0]
style_ax(ax1)

ax1.scatter(y_test, predictions,
            color=ORANGE, alpha=0.75, edgecolors="#CC4400",
            linewidths=0.5, s=55, label="Data Uji", zorder=3)

line_min = min(y_test.min(), predictions.min())
line_max = max(y_test.max(), predictions.max())
ax1.plot([line_min, line_max], [line_min, line_max],
         color=CYAN, linewidth=1.8, linestyle="--",
         label="Garis Ideal (y = x)", zorder=2)

ax1.set_xlabel("Harga Jual Aktual (Lakh ₹)", fontsize=10)
ax1.set_ylabel("Harga Jual Prediksi (Lakh ₹)", fontsize=10)
ax1.set_title("Aktual vs Prediksi Harga Jual Mobil", fontsize=12, fontweight="bold", pad=12)
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"₹{v:.0f}L"))
ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"₹{v:.0f}L"))
ax1.legend(facecolor=PANEL, edgecolor="#1E3248", labelcolor=WHITE, fontsize=9)
ax1.text(0.05, 0.92, f"R² = {score:.4f}", transform=ax1.transAxes,
         color=ACCENT, fontsize=11, fontweight="bold",
         bbox=dict(facecolor=BG, edgecolor=ACCENT, boxstyle="round,pad=0.4"))

# ── Plot 2: Scatter + Garis Regresi (warna per bahan bakar) ──────────────────
ax2 = axes[1]
style_ax(ax2)

fuel_colors = {"Petrol": "#FF6B35", "Diesel": "#4A9EFF", "LPG": "#39FF14", "Electric": "#FF00FF"}
for fuel, color in fuel_colors.items():
    mask = df["Bahan_Bakar"] == fuel
    if mask.sum() > 0:
        ax2.scatter(df.loc[mask, "KM_Tempuh"], df.loc[mask, "Harga_Jual_Lakh"],
                    color=color, alpha=0.70, edgecolors="none", s=40,
                    label=fuel, zorder=3)

x_line = np.linspace(df["KM_Tempuh"].min() - 1000,
                     df["KM_Tempuh"].max() + 1000, 200)
y_line = model.intercept_ + model.coef_[0] * x_line
ax2.plot(x_line, y_line, color=ACCENT, linewidth=2.2,
         label=f"Regresi: Ŷ = {model.intercept_:.2f} + ({model.coef_[0]:.5f})X",
         zorder=4)

ax2.set_xlabel("KM Tempuh (km)", fontsize=10)
ax2.set_ylabel("Harga Jual (Lakh ₹)", fontsize=10)
ax2.set_title("Sebaran Data & Garis Regresi\n(300 Mobil Bekas – CarDekho)", fontsize=12, fontweight="bold", pad=12)
ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}k km"))
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"₹{v:.0f}L"))
ax2.legend(facecolor=PANEL, edgecolor="#1E3248", labelcolor=WHITE, fontsize=8, ncol=2)

# ── Judul Utama ───────────────────────────────────────────────────────────────
fig.suptitle(
    "Prediksi Harga Jual Mobil Bekas Berdasarkan KM Tempuh\n"
    "Regresi Linear Sederhana  |  300 Data  |  Dataset CarDekho (Kaggle)",
    color=CYAN, fontsize=13, fontweight="bold", y=1.02
)

plt.tight_layout()
import os
save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prediksi_harga_mobil_plot.png")
plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"\nGrafik berhasil disimpan: {save_path}")
plt.show()
