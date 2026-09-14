################################################
# PREDICTION (backtest)
################################################
#
'''Eğitim verisinden rastgele bir yarış seçip modeli deniyor. Seçilen yarış
results_2010_2025_features.csv içinde hesaplanmış durumda olduğu için
 add_features çağrılmıyor.
'''
import joblib
import pandas as pd

from src.utils import f1_data_prep

def main():
    df = pd.read_csv("data/data_processed/results_2010_2025_features.csv")
    model = joblib.load("models/f1_model.pkl")

    # X'i TÜM veri üzerinden çıkarıyoruz, sadece seçilen yarışın satırlarından
    # değil. Sebebi: team_unified one-hot'a çevrilirken, sadece o yarıştaki
    # takımlar görülürse eksik sütun çıkar, model eğitimdeki sütunlarla
    # eşleşmez. Önce tüm veriyi encode edip sonra yarışı seçiyoruz.
    X, y = f1_data_prep(df)

    season, round_ = df[['season', 'round']].drop_duplicates().sample(1).iloc[0]
    race_idx = df[(df['season'] == season) & (df['round'] == round_)].index

    X_race = X.loc[race_idx]
    race_info = df.loc[race_idx, ['driver_id', 'team_unified', 'grid', 'position', 'is_podium']].copy()
    race_info['podium_prob'] = model.predict_proba(X_race)[:, 1]
    race_info = race_info.sort_values('podium_prob', ascending=False)

    predicted_top3 = race_info.head(3)['driver_id'].tolist()
    actual_top3 = race_info.loc[race_info['is_podium'] == 1, 'driver_id'].tolist()

    print(f"Secilen yaris: sezon {int(season)}, round {int(round_)}\n")
    print(race_info.to_string(index=False))
    print(f"\nModelin tahmini podyum: {predicted_top3}")
    print(f"Gercek podyum: {actual_top3}")

if __name__ == "__main__":
    main()


