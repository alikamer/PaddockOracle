# %%
import time

import pandas as pd
import requests

# %%
BASE_URL = "https://api.jolpi.ca/ergast/f1"
START_SEASON = 1991
END_SEASON = 2025
PAGE_LIMIT = 100
REQUEST_DELAY_SECONDS = 3
MAX_RETRIES = 5


# %%
def fetch_page(season: int, offset: int) -> dict:
    url = f"{BASE_URL}/{season}/results.json"
    params = {"limit": PAGE_LIMIT, "offset": offset}

    for attempt in range(1, MAX_RETRIES + 1):
        response = requests.get(url, params=params, timeout=30)

        # 429 = cok fazla istek. Her denemede bekleme suresi artiyor.
        if response.status_code == 429:
            wait_time = REQUEST_DELAY_SECONDS * attempt
            print(f"  429 alindi, {wait_time} sn beklenecek (deneme {attempt}/{MAX_RETRIES})")
            time.sleep(wait_time)
            continue

        response.raise_for_status()
        return response.json()

    raise RuntimeError(f"{season} sezonu offset={offset} icin {MAX_RETRIES} denemede basarisiz")


# %%
def fetch_season_results(season: int) -> list[dict]:
    all_races = []
    offset = 0

    # Sezonun kac sayfa oldugu onceden bilinmiyor; API'nin dondugu "total"
    # degerine ulasinca duruyoruz.
    while True:
        payload = fetch_page(season, offset)
        mrdata = payload["MRData"]
        all_races.extend(mrdata["RaceTable"]["Races"])

        offset += PAGE_LIMIT
        time.sleep(REQUEST_DELAY_SECONDS)

        if offset >= int(mrdata["total"]):
            break

    return all_races


# %%
def flatten_races_to_rows(races: list[dict]) -> list[dict]:
    rows = []

    for race in races:
        for result in race["Results"]:
            rows.append(
                {
                    "season": race["season"],
                    "round": race["round"],
                    "race_name": race["raceName"],
                    "circuit_id": race["Circuit"]["circuitId"],
                    "date": race["date"],
                    "driver_id": result["Driver"]["driverId"],
                    "constructor_id": result["Constructor"]["constructorId"],
                    "grid": result["grid"],
                    # position, yarisi bitiremeyenlerde hic gelmeyebiliyor:
                    # [] yerine .get() kullanmazsak KeyError aliriz.
                    "position": result.get("position"),
                    "position_text": result.get("positionText"),
                    "points": result["points"],
                    "laps": result["laps"],
                    "status": result["status"],
                }
            )

    return rows


# %%
def main() -> None:
    all_rows = []

    for season in range(START_SEASON, END_SEASON + 1):
        print(f"Cekiliyor: {season}")
        rows = flatten_races_to_rows(fetch_season_results(season))
        all_rows.extend(rows)
        print(f"  -> {len(rows)} satir (toplam: {len(all_rows)})")

    df = pd.DataFrame(all_rows)
    output_path = f"data/data_raw/results_{START_SEASON}_{END_SEASON}.csv"
    df.to_csv(output_path, index=False)
    print(f"Kaydedildi: {len(df)} satir -> {output_path}")


# %%
if __name__ == "__main__":
    main()
