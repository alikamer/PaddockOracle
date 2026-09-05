import os

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "data", "data_raw")
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "data_processed")

df = pd.read_csv(os.path.join(DATA_RAW_DIR, "results_all_seasons.csv"))

df_2010_plus = df[df["season"] >= 2010].reset_index(drop=True)

output_path = os.path.join(DATA_PROCESSED_DIR, "results_2010_2025.csv")
df_2010_plus.to_csv(output_path, index=False)

print(f"{len(df_2010_plus)} satir '{output_path}' dosyasina kaydedildi")
