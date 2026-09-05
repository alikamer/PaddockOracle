import os
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "data", "data_raw")

seasons_1950_1990 = pd.read_csv(os.path.join(DATA_RAW_DIR, "results_1950_1990.csv"))
seasons_1991_2025 = pd.read_csv(os.path.join(DATA_RAW_DIR, "results_1991_2025.csv"))

combined = pd.concat([seasons_1950_1990, seasons_1991_2025], ignore_index=True)

output_path = os.path.join(DATA_RAW_DIR, "results_all_seasons.csv")
combined.to_csv(output_path, index=False)

print(f"Birlesme tamamlandi: {len(seasons_1950_1990)} + {len(seasons_1991_2025)} = {len(combined)} satir")
print(f"Kaydedildi: {output_path}")

