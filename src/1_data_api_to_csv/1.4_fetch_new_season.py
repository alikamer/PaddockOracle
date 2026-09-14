################################################
# YENİ SEZONU ÇEKME (2026 ve sonrası)
################################################
# Çalıştırma (proje kökünden):  python src/1_data_api_to_csv/1.4_fetch_new_season.py
#
# Model 2025 sonuna kadar eğitildi. Yeni sezon AYRI bir dosyaya yazılıyor,
# eğitim verisine karışmıyor.
#
# Türetilmiş özellikler geçmişe baktığı için önce eski veriyle birleştirilip
# add_features çalıştırılıyor, sonra yalnızca yeni sezon ayrılıp kaydediliyor.

import time

import pandas as pd
import requests

from src.utils import add_features

BASE_URL = "https://api.jolpi.ca/ergast/f1"
SEASON = 2026
PAGE_LIMIT = 100          # API tek seferde en fazla 100 satır dönüyor
REQUEST_DELAY_SECONDS = 1

CLEAN_PATH = "data/data_processed/results_2010_2025_clean.csv"
OUTPUT_PATH = f"data/data_processed/results_{SEASON}_features.csv"
CALENDAR_PATH = f"data/data_processed/calendar_{SEASON}.csv"


def fetch_season(season):
    """Sezonun tüm yarış sonuçlarını sayfa sayfa çeker.

    API 100 satırda kesiyor; offset 'total' değerine ulaşana kadar artıyor.
    Aynı yarış birden fazla sayfaya bölünebilir.
    """
    rows = []
    offset = 0

    while True:
        response = requests.get(f"{BASE_URL}/{season}/results.json",
                                params={"limit": PAGE_LIMIT, "offset": offset},
                                timeout=30)
        response.raise_for_status()
        mrdata = response.json()["MRData"]

        for race in mrdata["RaceTable"]["Races"]:
            for result in race["Results"]:
                rows.append({
                    "season": int(race["season"]),
                    "round": int(race["round"]),
                    "race_name": race["raceName"],
                    "circuit_id": race["Circuit"]["circuitId"],
                    "date": race["date"],
                    "driver_id": result["Driver"]["driverId"],
                    "constructor_id": result["Constructor"]["constructorId"],
                    "grid": int(result["grid"]),
                    # Yarışı bitiremeyenlerde position gelmeyebiliyor
                    "position": result.get("position"),
                    "position_text": result.get("positionText"),
                    "points": result["points"],
                    "laps": result["laps"],
                    "status": result["status"],
                })

        offset += PAGE_LIMIT
        if offset >= int(mrdata["total"]):
            break
        time.sleep(REQUEST_DELAY_SECONDS)

    return pd.DataFrame(rows)


def fetch_calendar(season):
    """Sezonun tam takvimi: koşulmuş ve koşulmamış bütün yarışlar.

    Sonuç dosyası yalnızca koşulan yarışları içerir; koşulmamışları
    listeleyebilmek için takvim ayrıca gerekiyor.
    """
    response = requests.get(f"{BASE_URL}/{season}/races.json",
                            params={"limit": PAGE_LIMIT}, timeout=30)
    response.raise_for_status()
    races = response.json()["MRData"]["RaceTable"]["Races"]

    return pd.DataFrame([{
        "season": int(r["season"]),
        "round": int(r["round"]),
        "race_name": r["raceName"],
        "circuit_id": r["Circuit"]["circuitId"],
        "date": r["date"],
    } for r in races])


def clean(df):
    """3.1_data_cleaning.py'daki temizliğin yeni sezona uygulanmış hali."""
    # Start almamış pilotlar: hem 0 tur hem "W" koşulu birlikte aranır.
    # Tek başına "W" yarışı koşup sonra çekilenleri de silerdi.
    no_start = (df["laps"].astype(int) == 0) & (df["position_text"] == "W")
    df = df[~no_start].reset_index(drop=True)
    print(f"  start almamış satır silindi: {int(no_start.sum())}")

    # grid == 0 = pit lane start. Sıfır sayısal olarak pole gibi görünüyor,
    # oysa gridin en gerisi.
    grid_zero = df["grid"] == 0
    if grid_zero.any():
        max_grid = df.set_index(["season", "round"]).index.map(
            df[~grid_zero].groupby(["season", "round"])["grid"].max()
        )
        df.loc[grid_zero, "grid"] = (
            max_grid[grid_zero.to_numpy()]
            + df[grid_zero].groupby(["season", "round"]).cumcount()
            + 1
        )
    print(f"  grid = 0 düzeltildi: {int(grid_zero.sum())}")

    df["position"] = pd.to_numeric(df["position"])
    df["points"] = pd.to_numeric(df["points"])
    df["laps"] = pd.to_numeric(df["laps"])
    return df


def main():
    print(f"{SEASON} sezonu çekiliyor...")
    new = fetch_season(SEASON)
    print(f"  {len(new)} satır, {new['round'].nunique()} yarış")

    new = clean(new)

    old = pd.read_csv(CLEAN_PATH)
    print(f"Eğitim verisi: {len(old)} satır ({old['season'].min()}-{old['season'].max()})")

    # Birleşik tablo üzerinde hesaplanıyor ki yeni sezonun formları eski
    # sezonların sonuna bakabilsin.
    combined = pd.concat([old, new], ignore_index=True)
    combined["is_podium"] = (combined["position"] <= 3).astype(int)
    combined = add_features(combined)

    out = combined[combined["season"] == SEASON].reset_index(drop=True)
    out.to_csv(OUTPUT_PATH, index=False)

    print(f"Kaydedildi: {len(out)} satır, {out.shape[1]} sütun -> {OUTPUT_PATH}")
    print(f"Yarış sayısı: {out['round'].nunique()}")

    calendar = fetch_calendar(SEASON)
    calendar["done"] = calendar["round"].isin(out["round"].unique())
    calendar.to_csv(CALENDAR_PATH, index=False)
    print(f"Takvim kaydedildi: {len(calendar)} yarış "
          f"({int(calendar['done'].sum())} koşuldu) -> {CALENDAR_PATH}")


if __name__ == "__main__":
    main()
