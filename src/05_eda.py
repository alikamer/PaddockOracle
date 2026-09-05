"""Kesifci veri analizi (EDA) - calisma tablosu: data_processed/results_2010_2025.csv"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)
sns.set_theme(style="whitegrid")

df = pd.read_csv("data/data_processed/results_2010_2025.csv")

# ---- ilk bakis ----
print(df.info())
print(df.head())
print(df.isnull().sum())
print(df.duplicated().sum())

# ---- hedef degisken ----
df["podium"] = df["position"].isin([1, 2, 3]).astype(int)

# ---- betimsel istatistikler ----
print(df.describe())
print(df["status"].value_counts())

# ---- kategorik sutunlar ----
print("driver_id nunique:", df["driver_id"].nunique())
print("constructor_id nunique:", df["constructor_id"].nunique())
print("circuit_id nunique:", df["circuit_id"].nunique())
print(df["driver_id"].value_counts().head(10))
print(df["constructor_id"].value_counts().head(10))


# ---- outlier esikleri ----
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

# ---- gorseller ----

# Grid pozisyonuna gore podyuma girme orani
podium_by_grid = df[df["grid"] > 0].groupby("grid")["podium"].mean().reset_index()

plt.figure(figsize=(10, 6))
sns.lineplot(data=podium_by_grid, x="grid", y="podium", marker="o")
plt.title("Grid Pozisyonuna Gore Podyuma Girme Orani (2010+)")
plt.xlabel("Grid Pozisyonu")
plt.ylabel("Podyuma Girme Orani")
plt.show()

# Sayisal degiskenler arasi korelasyon
"""
Grid ve Position (0.41): pozitif iliski - sıralamada geride baslarsan yarisi da geride
bitirme ihtimalin artar. Grid, yarış sonucunu tahmin etmede güçlü bir girdi.
Position ve Points (-0.61): guclu ters orantı - bitirme sıran büyüdükce (geriye gidince)
kazandigin puan hizla duser.
Position ve Laps (-0.59): guclu ters orantı - az tur atabildiysen (dusuk laps) yarisi
genelde geride bitirmissindir (kaza/DNF).
Points ve Podium (0.70): en guclu dogru orantilardan biri - ilk 3'e girmek devasa puan
getirir, bu iki degisken el ele hareket eder.
"""
numeric_cols = ["grid", "position", "points", "laps", "podium"]
correlation_matrix = df[numeric_cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
plt.title("Sayisal Degiskenler Arasi Korelasyon (2010+)")
plt.show()

# Podyum vs podyum olmayanlarin grid dagilimi
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x="podium", y="grid")
plt.title("Podyum Durumuna Gore Grid Pozisyonu Dagilimi (2010+)")
plt.xlabel("Podyum (0 = Hayir, 1 = Evet)")
plt.ylabel("Grid Pozisyonu")
plt.show()

# En sik gorulen bitis durumlari (top 10)
plt.figure(figsize=(10, 6))
df["status"].value_counts().head(10).plot(kind="barh")
plt.title("En Sik Gorulen Bitis Durumlari (Top 10, 2010+)")
plt.xlabel("Satir Sayisi")
plt.gca().invert_yaxis()
plt.show()
