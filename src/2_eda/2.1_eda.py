# %%
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)
sns.set_theme(style="whitegrid")

# %%
# Temizlenmemis dosya okunuyor: kirlilikler burada tespit edilip 3.1'de gideriliyor.
df = pd.read_csv("data/data_processed/results_2010_2025.csv")

# %%
# ---- ilk bakis ----
print(df.info())
print(df.head())
print(df.isnull().sum())
print("tekrar eden satir:", df.duplicated().sum())

# %%
# ---- hedef degisken ----
df["podium"] = df["position"].isin([1, 2, 3]).astype(int)
print(df["podium"].value_counts())
print("podyum orani:", round(df["podium"].mean(), 4))

# %%
# ---- betimsel istatistikler ----
print(df.describe())
print(df["status"].value_counts())

# %%
# ---- kategorik sutunlar ----
for col in ["driver_id", "constructor_id", "circuit_id"]:
    print(f"{col} nunique: {df[col].nunique()}")

print(df["driver_id"].value_counts().head(10))
print(df["constructor_id"].value_counts().head(10))


# %%
# ---- aykiri deger esikleri ----
def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    return dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None)


for col in ["grid", "laps", "points"]:
    print(col, check_outlier(df, col))

# %%
# ---- grid pozisyonuna gore podyum orani ----
# grid == 0 (pit lane start) haric tutuluyor; 3.1'de duzeltiliyor.
podium_by_grid = df[df["grid"] > 0].groupby("grid")["podium"].mean().reset_index()

plt.figure(figsize=(10, 6))
sns.lineplot(data=podium_by_grid, x="grid", y="podium", marker="o")
plt.title("Grid Pozisyonuna Gore Podyuma Girme Orani (2010+)")
plt.xlabel("Grid Pozisyonu")
plt.ylabel("Podyuma Girme Orani")
plt.show()

# %%
# ---- korelasyon ----
# Bulgu: grid-position (0.41) onden baslamak one bitirmeyi getiriyor.
# points/laps ile podium arasindaki guclu iliski yaris SONRASI bilgi oldugu
# icin modele girmeyecek - burada sadece dogrulama amacli.
numeric_cols = ["grid", "position", "points", "laps", "podium"]

plt.figure(figsize=(8, 6))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
plt.title("Sayisal Degiskenler Arasi Korelasyon (2010+)")
plt.show()

# %%
# ---- podyum vs grid dagilimi ----
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x="podium", y="grid")
plt.title("Podyum Durumuna Gore Grid Pozisyonu Dagilimi (2010+)")
plt.xlabel("Podyum (0 = Hayir, 1 = Evet)")
plt.ylabel("Grid Pozisyonu")
plt.show()

# %%
# ---- en sik bitis durumlari ----
plt.figure(figsize=(10, 6))
df["status"].value_counts().head(10).plot(kind="barh")
plt.title("En Sik Gorulen Bitis Durumlari (Top 10, 2010+)")
plt.xlabel("Satir Sayisi")
plt.gca().invert_yaxis()
plt.show()
