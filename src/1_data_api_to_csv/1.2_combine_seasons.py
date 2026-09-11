# %%
import pandas as pd

# %%
old = pd.read_csv("data/data_raw/results_1950_1990.csv")
new = pd.read_csv("data/data_raw/results_1991_2025.csv")

# %%
combined = pd.concat([old, new], ignore_index=True)
print(f"{len(old)} + {len(new)} = {len(combined)} satir")

# %%
output_path = "data/data_raw/results_all_seasons.csv"
combined.to_csv(output_path, index=False)
print(f"Kaydedildi: {output_path}")
