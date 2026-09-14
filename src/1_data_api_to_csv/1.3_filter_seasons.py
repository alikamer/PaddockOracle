# %%
import pandas as pd

# %%
df = pd.read_csv("data/data_raw/results_all_seasons.csv")
print(f"Girdi: {len(df)} satir, {df.season.min()}-{df.season.max()}")

# %%
# 2010 kesme noktası: öncesinde puanlama sistemi ve teknik kurallar çok farklı,
# eski sezonlar modele gürültü katıyor.
df = df[df["season"] >= 2010].reset_index(drop=True)
print(f"2010+ : {len(df)} satir")

# %%
output_path = "data/data_processed/results_2010_2025.csv"
df.to_csv(output_path, index=False)
print(f"Kaydedildi: {output_path}")
