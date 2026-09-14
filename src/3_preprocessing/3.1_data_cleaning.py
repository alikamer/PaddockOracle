# %%
import pandas as pd

# %%
df = pd.read_csv("data/data_processed/results_2010_2025.csv")
print(f"Girdi: {len(df)} satir")








# %%
# Start almamış pilotlar. İki koşul birlikte aranmalı: tek başına
# status == "Withdrew" yarışı koşup sonra çekilenleri de yakalıyor,
# tek başına laps == 0 ise 1. turda kaza yapıp çıkanları yakalıyor.


no_start = (df["laps"] == 0) & (df["position_text"] == "W")
df = df[~no_start].reset_index(drop=True)
print(f"Start almamis satirlar silindi: {no_start.sum()}")








# %%
# grid == 0 = pit lane start. Sıfır sayısal olarak pole'dan iyi görünüyor,
# oysa gridin en gerisi: o yarıştaki en yüksek grid değerinin arkasına diziliyor.
grid_zero = df["grid"] == 0
max_grid = df.set_index(["season", "round"]).index.map(
    df[~grid_zero].groupby(["season", "round"])["grid"].max()
)
df.loc[grid_zero, "grid"] = (
    max_grid[grid_zero.to_numpy()]
    + df[grid_zero].groupby(["season", "round"]).cumcount()
    + 1
)
print(f"grid = 0 duzeltildi: {grid_zero.sum()}")







# %%
podium_per_race = df.groupby(["season", "round"])["position"].apply(lambda s: (s <= 3).sum())

assert df.isnull().sum().sum() == 0, "bos hucre"
assert df.duplicated(subset=["season", "round", "driver_id"]).sum() == 0, "ayni yariste ayni pilot"
assert (df["grid"] > 0).all(), "grid'de sifir kaldi"
assert (podium_per_race == 3).all(), "yarista 3 podyum yok"

print(f"Kontroller gecti | {len(df)} satir, {len(podium_per_race)} yaris")






# %%
output_path = "data/data_processed/results_2010_2025_clean.csv"
df.to_csv(output_path, index=False)
print(f"Kaydedildi: {output_path}")
