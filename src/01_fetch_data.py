import os
import time #api'ye çok fazla istek atmamak için sleep gerekli

import pandas as pd #csv
import requests #HTTP isteği atıp API'dne JSON cevap almamızı sağlayan lib


#CONSTANTS
BASE_URL = "https://api.jolpi.ca/ergast/f1"
START_SEASON = 1991 #Modelde 2010() veya 2014 kullanılacak  atmosferik/hibrit motor geçişi
END_SEASON = 2025
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGE_LIMIT = 100
REQUEST_DELAY_SECONDS = 3
MAX_RETRIES = 5


'''
Satır satır ne oluyor:
- def fetch_page(season: int, offset: int) -> dict: → fonksiyon iki parametre alıyor: hangi sezon, kaçıncı kayıttan başlasın (offset). -> dict kısmı fonksiyonun bir sözlük (JSON'dan gelen veri) döndüreceğini söylüyor — bu bir "type hint", zorunlu değil ama okuyana ne döneceğini söylüyor.
- url = f"{BASE_URL}/{season}/results.json" → f-string ile URL'i birleştiriyoruz. Örneğin season=2020 ise https://api.jolpi.ca/ergast/f1/2020/results.json olur.
- params = {...} → requests kütüphanesi bu sözlüğü otomatik olarak URL'in sonuna ?limit=100&offset=0 şeklinde ekliyor, sen elle string birleştirmiyorsun.
- requests.get(url, params=params, timeout=30) → isteği atıyor. timeout=30 önemli: API 30 saniyede cevap vermezse kod sonsuza kadar takılı kalmasın diye bir hata fırlatıyor.
- response.raise_for_status() → eğer API 404/500 gibi bir hata kodu dönerse, burada anlamlı bir hata fırlatır. Bu satır olmasaydı, hatalı bir cevabı fark etmeden .json() çağırıp anlamsız bir hatayla karşılaşabilirdik.
- return response.json() → gelen cevabı Python sözlüğüne çevirip döndürüyor.
'''
def fetch_page(season: int, offset: int) -> dict:
    url=f"{BASE_URL}/{season}/results.json"
    params = {"limit": PAGE_LIMIT, "offset": offset}

    for attempt in range(1, MAX_RETRIES + 1):
        response = requests.get(url, params=params, timeout=30)

        if response.status_code == 429:
            wait_time = REQUEST_DELAY_SECONDS * attempt
            print(f"  429 alindi, {wait_time} saniye beklenip tekrar denenecek (deneme {attempt}/{MAX_RETRIES})")
            time.sleep(wait_time)
            continue

        response.raise_for_status()
        return response.json()

    raise RuntimeError(f"{season} sezonu offset={offset} icin {MAX_RETRIES} denemede basarili olunamadi")



'''
Satır satır:
- all_races = [], offset = 0 → boş bir liste ve sayaç ile başlıyoruz. Bu sezona ait tüm yarışları burada biriktireceğiz.
- while True: → şartsız sonsuz döngü. Normalde tehlikeli görünür ama içeride break ile kendimiz durduruyoruz — kaç sayfa olduğunu önceden bilmediğimiz için for yerine bunu kullanıyoruz.
- payload = fetch_page(season, offset) → az önce yazdığımız fonksiyonu çağırıp o sayfanın JSON'unu alıyoruz.
- mrdata = payload["MRData"] → API'nin cevabı hep MRData anahtarının içinde geliyor (Jolpica'nın/Ergast'ın sabit zarfı).
- races = mrdata["RaceTable"]["Races"] → o sayfadaki yarışların listesi.
- all_races.extend(races) → append değil extend kullandık çünkü races zaten bir liste, onu tek tek elemanlarıyla ana listeye ekliyoruz (append kullansaydık liste içinde liste olurdu).
- total = int(mrdata["total"]) → API bize "bu sezonda toplam kaç sonuç satırı var" bilgisini string olarak veriyor, int() ile sayıya çeviriyoruz.
- offset += PAGE_LIMIT → bir sonraki sayfaya geçmek için offset'i 100 artırıyoruz.
- time.sleep(REQUEST_DELAY_SECONDS) → API'yi yormamak için 1 saniye bekliyoruz, her sayfa isteğinden sonra.
- if offset >= total: break → eğer artık aldığımız kayıt sayısı toplamı geçtiyse (ya da eşitse), döngüden çık.

Neden season yerine burada all_races biriktiriyoruz, fetch_page'de değil? Çünkü fetch_page sadece "bir sayfa getir" işini biliyor, "bir sezonun tamamını topla" mantığı ayrı bir sorumluluk — her fonksiyon tek bir işi yapsın diye ayırdık.
'''
def fetch_seaon_results(season: int) -> list[dict]:
    all_races=[]
    offset = 0

    while True:
        payload = fetch_page(season,offset)
        mrdata = payload['MRData']
        races = mrdata["RaceTable"]["Races"]
        all_races.extend(races)

        total = int(mrdata["total"])
        offset += PAGE_LIMIT
        time.sleep(REQUEST_DELAY_SECONDS)

        if offset >= total:
            break #offset totalı aşarsa döngü kırılmalıdır aksi halde sonsuz döner

    return all_races



#flatten_races_to_rows exp
'''
Satır satır:
- for race in races: → dışta gelen yarışları tek tek geziyoruz.
- for result in race["Results"]: → her yarışın içindeki, o yarışa katılan her pilotun sonucunu tek tek geziyoruz. İki döngü iç içe olduğu için sonunda toplam satır sayısı = yarış sayısı × yarış başına pilot sayısı olacak.
- rows.append({...}) → her pilot-yarış kombinasyonu için bir sözlük oluşturup listeye ekliyoruz. Bu sözlüğün anahtarları, sonunda tablonun sütun adları olacak.
- race["season"], race["round"] gibi alanlar → yarışın kendi bilgisi, o yarışın tüm sonuçlarında aynı kalır (sezon, tur no, yarış adı, pist, tarih).
- result["Driver"]["driverId"], result["Constructor"]["constructorId"] → pilot ve takımın kimliği, iç içe bir sözlükten çekiliyor (Driver altında ayrı bir sözlük var, içinde driverId var).
- result["grid"] → sıralama pozisyonu (grid), result.get("position") ve result.get("positionText") → bitiş pozisyonu. Burada [...] yerine .get(...) kullandık çünkü yarışı bitirmeyen (DNF) pilotlarda position alanı hiç gelmeyebilir — .get() öyle bir durumda hata vermez, None döndürür. [...] kullansaydık, o pilotta KeyError alırdık ve kod çökerdi.
- result["points"], result["laps"], result["status"] → puan, tamamlanan tur sayısı, yarış durumu (bitirdi mi, arıza mı vs.) — bunlar her sonuçta garanti var, o yüzden direkt [...] kullandık.
'''
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
                    "position": result.get("position"),
                    "position_text": result.get("positionText"),
                    "points": result["points"],
                    "laps": result["laps"],
                    "status": result["status"],
                }
            )

    return rows




'''
Satır satır:
- all_rows = [] → tüm sezonların tüm satırlarını burada toplayacağız.
- for season in range(START_SEASON, END_SEASON + 1): → 1950'den 2025'e kadar dönüyoruz. +1 şart, çünkü range son sayıyı dahil etmez (range(1950, 2025) 2024'te biterdi).
- print(f"Cekiliyor: {season} sezonu") → 75 sezon boyunca sessizce beklemek yerine hangi sezonda olduğumuzu ekrana yazdırıyoruz — ilerlemeyi takip edebilmen için, ayrıca bir yerde takılırsa hangi sezonda takıldığını görürsün.
- races = fetch_season_results(season) → o sezonun tüm yarışlarını (sayfalanmış halde) çekiyoruz.
- rows = flatten_races_to_rows(races) → düz satırlara çeviriyoruz.
- all_rows.extend(rows) → ana listeye ekliyoruz.
- ikinci print → o sezonda kaç satır geldi, toplamda şu ana kadar kaç satır birikti — ilerleme takibi için.
- df = pd.DataFrame(all_rows) → tüm satırların listesini (her biri sözlük) pandas tablosuna çeviriyoruz. Sözlüklerin anahtarları otomatik sütun adı oluyor.
- df.to_csv(output_path, index=False) → CSV'ye kaydediyoruz. index=False önemli: yoksa pandas soldan başlayan 0,1,2,... satır numarası sütununu da CSV'ye yazar, bize gereksiz bir sütun eklemiş olur.
- if __name__ == "__main__": main() → bu dosyayı doğrudan çalıştırırsan (python fetch_jolpica_data.py) main() çağrılsın, ama ileride bu dosyadan bir fonksiyonu başka bir dosyaya import edersen (mesela sadece fetch_page'i kullanmak istersen), main() otomatik çalışmasın diye bu kontrol var. Küçük bir Python konvansiyonu, her dosyanın başında görürsün.
''' #main exp
def main() -> None:
    all_rows = []

    for season in range(START_SEASON,END_SEASON+1):  #range dahil etmiyordu sonu
        print(f'Cekiliyor: {season} sezonu')
        races = fetch_seaon_results(season)
        rows = flatten_races_to_rows(races)
        all_rows.extend(rows)
        print(f'-> {len(rows)} satir eklendi (toplam: {len(all_rows)})')

    df = pd.DataFrame(all_rows)
    output_filename = f'results_{START_SEASON}_{END_SEASON}.csv'
    output_path = os.path.join(PROJECT_ROOT, 'data', 'data_raw', output_filename)
    df.to_csv(output_path, index=False)
    print(f"Bitti. {len(df)} satir '{output_path}' dosyasina kaydedildi.")

if __name__ == "__main__":    #kontrol kısmı önemli, sabit kalsın
    main()




