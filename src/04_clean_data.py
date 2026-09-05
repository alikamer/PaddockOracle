import os

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "data_processed")

df = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "results_2010_2025.csv"))

# "Withdrew" = pilot yarisa hic katilmadi, gercek bir sonuc degil, atiliyor
before = len(df)
df = df[df["status"] != "Withdrew"].reset_index(drop=True)
print(f"Withdrew satirlari silindi: {before - len(df)} satir")

# grid == 0 = pilot yarisa katildi ama pit lane'den basladi (kural ihlali, teknik ariza vb.)
# gercekte gridin en gerisinde sayilir: o yaristaki en yuksek gercek grid degerinden
# devam ettiriliyor. Ayni yariste birden fazla pit lane start varsa, orijinal satir
# sirasina gore ardisik degerler veriliyor (aralarindaki gercek sira bilinmiyor)
grid_zero_mask = df["grid"] == 0
max_real_grid_per_race = df[~grid_zero_mask].groupby(["season", "round"])["grid"].max()
max_real_grid = df.set_index(["season", "round"]).index.map(max_real_grid_per_race)
zero_rank = df[grid_zero_mask].groupby(["season", "round"]).cumcount() + 1
df.loc[grid_zero_mask, "grid"] = max_real_grid[grid_zero_mask.to_numpy()] + zero_rank
print(f"grid=0 duzeltildi: {grid_zero_mask.sum()} satir")

output_path = os.path.join(DATA_PROCESSED_DIR, "results_2010_2025_clean.csv")
df.to_csv(output_path, index=False)
print(f"{len(df)} satir '{output_path}' dosyasina kaydedildi")
