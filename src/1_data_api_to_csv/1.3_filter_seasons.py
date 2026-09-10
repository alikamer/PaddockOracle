# %%
import pandas as pd

# %%
df = pd.read_csv("data/data_raw/results_all_seasons.csv")
print(f"Girdi: {len(df)} satir, {df.season.min()}-{df.season.max()}")

# %%
# 2010 kesme noktasi: oncesinde puanlama sistemi ve teknik kurallar cok farkli,
# eski sezonlar modele gurultu katiyor.
df = df[df["season"] >= 2010].reset_index(drop=True)
print(f"2010+ : {len(df)} satir")

# %%
output_path = "data/data_processed/results_2010_2025.csv"
df.to_csv(output_path, index=False)
print(f"Kaydedildi: {output_path}")
