# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %%

df = pd.read_csv("data/data_processed/results_2010_2025_clean.csv")
print(f"Girdi: {len(df)} satir")
df.head()
pd.set_option('display.max_columns',None)
pd.set_option('display.width',500)

# Hedef degisken: podyuma girdi mi (ilk 3)
df["is_podium"] = (df["position"] <= 3).astype(int)


# %%
# Yardimci fonksiyonlar — her yeni feature'dan sonra cagrilir.
# Hesap yapan fonksiyonlar degil, sadece bakis/dogrulama araclari.

sns.set_theme(style="whitegrid")  # seaborn'un hazir temasi: silik izgara, ince cerceve


def feature_summary(dataframe, col, target="is_podium", bins=5, plot=False):
    """

    Yeni turetilen bir feature'in ozeti: bosluk orani, dagilim ve hedefle iliskisi.

    Parameters
    ------
        dataframe: dataframe
                Incelenecek dataframe
        col: str
                Incelenecek sayisal feature
        target: str, optional
                Hedef sutun (0/1). Varsayilan "is_podium".
        bins: int, optional
                Feature kac esit buyuklukte gruba bolunecek (qcut)
        plot: bool, optional
                True ise grup bazli hedef oranini bar grafigi olarak da cizer

    Notes
    ------
        Gecmise bakan bir feature'in ilk satirlarinda bosluk OLMAK ZORUNDA.
        "Bos deger: 0" goruyorsan shift(1) unutulmus demektir — sessiz sizinti alarmi.

    """

    na_count = dataframe[col].isnull().sum()

    print(f"##################### {col} #####################")
    print(f"Bos deger: {na_count} ({100 * na_count / len(dataframe):.1f}%)")
    print(dataframe[col].describe([0.05, 0.25, 0.50, 0.75, 0.95]).T)

    binned = dataframe.dropna(subset=[col]).copy()
    binned["_bin"] = pd.qcut(binned[col], bins, duplicates="drop", precision=1)
    rates = binned.groupby("_bin", observed=True)[target].mean()

    print(f"##################### {target} orani #####################")
    print(rates)

    if plot:
        rates.plot(kind="bar", color="steelblue", figsize=(9, 5))
        plt.xlabel(col)
        plt.ylabel(f"{target} orani")
        plt.title(f"{col} araligina gore {target} orani")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.show()


def feature_vs_position(dataframe, col):
    # Dagilim + regresyon dogrusu: feature buyudukce bitis pozisyonu ne oluyor?
    plt.figure(figsize=(8, 6))
    sns.regplot(
        data=dataframe.dropna(subset=[col]),
        x=col,
        y="position",
        scatter_kws={"alpha": 0.15, "s": 15},
        line_kws={"color": "red"},
    )
    plt.gca().invert_yaxis()
    plt.xlabel(col)
    plt.ylabel("Gercek bitis pozisyonu")
    plt.title(f"{col} gercek sonucu isaret ediyor mu?")
    plt.tight_layout()
    plt.show()


def correlation_matrix(dataframe, cols):
    plt.gcf().set_size_inches(10, 8)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    sns.heatmap(dataframe[cols].corr(), annot=True, linewidths=0.5, vmin=-1, vmax=1,
                annot_kws={"size": 12}, linecolor="w", cmap="RdBu")
    plt.tight_layout()
    plt.show()


def driver_timeline(dataframe, driver_id, cols):
    # Sadece pozisyon olcegindeki sutunlar icin anlamli (rolling_form gibi).
    driver = dataframe[dataframe["driver_id"] == driver_id]

    plt.figure(figsize=(14, 5))
    plt.plot(range(len(driver)), driver["position"], label="Gercek bitis pozisyonu",
             marker="o", alpha=0.4)
    for col in cols:
        plt.plot(range(len(driver)), driver[col], label=col, linewidth=2)

    plt.gca().invert_yaxis()
    plt.xlabel("Kariyer sirasi (yaris no)")
    plt.ylabel("Pozisyon")
    plt.title(f"{driver_id}: gercek pozisyon vs turetilen ozellikler")
    plt.legend()
    plt.tight_layout()
    plt.show()


# %%
# Takim ismi birlestirme (toro_rosso->alphatauri->rb gibi zincirler)

#Burada ufak bir sorunumuz var düzeltmemiz gerekmekte
#constructor_id isimli sütunda takımın marka bilgisi tutulmakta, fakat yıllar içerisinde bazı takımlar bazı sebeplerden(satılma,sponsor vs) ötürü isim değiştirmiş oysa ki takım aynı takım
#Bu noktada  takım isimlerini birleştirmemiz gerekiyor
'''
┌───────────────────────────────────────────┬───────────────────────────────────────────────┐
│                  Zincir                   │                    Yıllar                     │
├───────────────────────────────────────────┼───────────────────────────────────────────────┤
│ toro_rosso → alphatauri → rb              │ 2010-19 → 20-23 → 24-25                       │
├───────────────────────────────────────────┼───────────────────────────────────────────────┤
│ force_india → racing_point → aston_martin │ 2010-18 → 19-20 → 21-25                       │
├───────────────────────────────────────────┼───────────────────────────────────────────────┤
│ renault → lotus_f1 → renault → alpine     │ 2010-11 → 12-15 → 16-20 → 21-25               │
├───────────────────────────────────────────┼───────────────────────────────────────────────┤
│ sauber → alfa → sauber                    │ 2010-18 → 19-23 → 24-25                       │
├───────────────────────────────────────────┼───────────────────────────────────────────────┤
│ lotus_racing → caterham                   │ 2010-11 → 12-14 (sonra takım kapandı)         │
├───────────────────────────────────────────┼───────────────────────────────────────────────┤
│ virgin → marussia → manor                 │ 2010-11 → 12-14 → 15-16 (sonra takım kapandı) │
└───────────────────────────────────────────┴───────────────────────────────────────────────┘
'''
#ferrari, mclaren, mercedes, red_bull, williams, haas, hrt, Bu takımlar tarih boyunca ( en azından 2010-25) aralığında isim değiştirmemiş, raw olarak bırakılabilir

# Adim 1
#Nasıl çözeceğiz bu sorunu?
'''

Ne yapacağız: Bir sözlük (dictionary) kuracağız — her eski ismi zincirin tek bir ortak kimliğine eşleyeceğiz (mesela hepsini zincirin ilk ismine, ya da hepsini en güncel ismine — bunu birazdan karar vereceğiz).
Sonra df üzerinde yeni bir sütun açacağız, mesela team_unified, ve bu sütunu takım formu / takım arkadaşı farkı gibi geçmişe bakan hesaplarda kullanacağız.
Orijinal constructor_id sütununu silmeyeceğiz — o kalacak, ileride başka bir yerde (mesela son yarıştaki gerçek marka bilgisi olarak) işe yarayabilir.
'''


#team_unified sütununu ekleyelim df'e

team_name_map = {
    'toro_rosso': 'rb',
    'alphatauri': 'rb',
    'force_india': 'aston_martin',
    'racing_point': 'aston_martin',
    'renault': 'alpine',
    'lotus_f1': 'alpine',
    'alfa': 'sauber',
    'lotus_racing':'caterham',
    'virgin':'manor',
    'marussia':'manor',
}



# Adim 2
#sözlüğü uygulayıp team_unified sütununu açalım
'''kodun işlevi
Adım 1 — df['constructor_id']
Bu, tablodaki constructor_id sütununun tamamı — her satırda bir takım ismi var. Örnek birkaç satır düşün:

┌───────┬────────────────┐
│ satır │ constructor_id │
├───────┼────────────────┤
│ 1     │ toro_rosso     │
├───────┼────────────────┤
│ 2     │ ferrari        │
├───────┼────────────────┤
│ 3     │ caterham       │
├───────┼────────────────┤
│ 4     │ lotus_racing   │
└───────┴────────────────┘

Adım 2 — .map(team_name_map)
Bu, o sütundaki her değeri tek tek sözlükte arıyor, "bu değer sözlükte anahtar (key) olarak var mı" diye bakıyor:
- toro_rosso → sözlükte var → rb yazılır.
- ferrari → sözlükte yok → sonuç NaN (boş) olur.
- caterham → sözlükte yok (çünkü caterham bir değer/value, anahtar/key değil) → NaN.
- lotus_racing → sözlükte var → caterham yazılır.

Bu adımdan sonra tablo geçici olarak şöyle:

┌───────┬────────────┐
│ satır │ map sonucu │
├───────┼────────────┤
│ 1     │ rb         │
├───────┼────────────┤
│ 2     │ NaN        │
├───────┼────────────┤
│ 3     │ NaN        │
├───────┼────────────┤
│ 4     │ caterham   │
└───────┴────────────┘

Görüyorsun, ferrari ve caterham gibi zaten doğru/sabit olanlar burada boş kaldı — çünkü .map() sadece sözlükte bulduğunu yazar, bulamadığını boş bırakır, "olduğu gibi kalsın" demez.

Adım 3 — .fillna(df['constructor_id'])
İşte bu adım o boşlukları dolduruyor: "nerede NaN varsa, oraya orijinal constructor_id sütunundaki değeri koy" diyor. Yani:
- Satır 2 (NaN) → orijinali ferrari idi → ferrari yazılır.
- Satır 3 (NaN) → orijinali caterham idi → caterham yazılır.
- Satır 1 ve 4 zaten doluydu (rb, caterham), onlara dokunulmaz.

Sonuç (team_unified sütunu):

┌───────┬────────────────┬──────────────┐
│ satır │ constructor_id │ team_unified │
├───────┼────────────────┼──────────────┤
│ 1     │ toro_rosso     │ rb           │
├───────┼────────────────┼──────────────┤
│ 2     │ ferrari        │ ferrari      │
├───────┼────────────────┼──────────────┤
│ 3     │ caterham       │ caterham     │
├───────┼────────────────┼──────────────┤
│ 4     │ lotus_racing   │ caterham     │
└───────┴────────────────┴──────────────┘
'''
df['team_unified'] = df['constructor_id'].map(team_name_map).fillna(df['constructor_id'])







# %%
# 1. ROLLING FORM
'''
Rolling Form bir pilotun içinde bulunduğu yarıştan önceki birkaç (3 veya 5 diyelim) yarışta nasıl performans gösterdiğinin ortalamasıdır. Bu bize ne kadar formda olduğunu gösteren yeni bir feature sağlar. 
İlgili eğitim modeli, pilotu sabit verilerle eğitip tahmin etmek yerine  dinamik formunu göz önünde bulundurur.
Formula 1 gibi bir sporda da form çok önemli bir parametredir(bağımsız değişkendir).
Dolayısıyla bunu feature engineering ile üretip modele kazandırmamız elzemdir

'''

# Adim 1
# Elimizdeki tablo şu an satır satır (her yarış-pilot kombinasyonu bir satır) duruyor ama kronolojik sırada değil — CSV'de hangi sırada geldiyse öyle duruyor.
# "Son 3 yarış" hesaplayabilmek için önce her pilotun kendi yarışlarını zaman sırasına göre dizmemiz lazım.
'''
Adım 1 — Sıralama: df'i driver_id, sonra season, sonra round'a göre sıralayacağız. Böylece her pilotun kendi satırları art arda ve kronolojik sırada gelecek.

Adım 2 — Gruplama: driver_id'ye göre gruplayacağız — her pilotun geçmişi sadece kendi geçmişinden hesaplanmalı, başka pilotun yarışları karışmamalı.

Adım 3 — Shift (kaydırma) — en kritik kısım: Eğer doğrudan "son 3 yarışın ortalaması" hesaplarsak, o anki yarışın kendisi de hesaba karışır — bu sızıntı olur (o yarışın sonucunu, o yarışı tahmin ederken kullanmış oluruz).
Bu yüzden önce diziyi bir yarış geriye kaydırıyoruz (shift(1)): "şu anki satırın karşısına, kendisinden önceki satırın değerini koy" diyoruz.
Sonra o kaydırılmış dizinin üzerinden 3'lük pencere ortalaması alıyoruz.
'''

df = df.sort_values(['driver_id', 'season', 'round']).reset_index(
    drop=True)  # satır indexlerini resetliyor sıralama sonrası, nizami dursun diye yoksa bir olayı yok


for w in [3, 4, 5, 8]:
    test = df.groupby('driver_id')['position'].transform(lambda s: s.shift(1).rolling(w).mean())
    print(f"pencere {w}: korelasyon {test.corr(df['is_podium']):.3f} | bos satir {test.isna().sum()}")


df['rolling_form'] = (
    df.groupby('driver_id')['position']
    .transform(lambda s: s.shift(1).rolling(3).mean())  # shift(1) data leakage 'i önlemek için çok önemli !!!

)

#Peki neden 3 yarış? Neden 4, 5 ya da 8 değil?
'''
 Dört farklı pencereyi kurup her birinin podyumla
korelasyonuna ve kaç satırı boşa düşürdüğüne baktık:

┌─────────┬─────────────────────────┬──────────────┐
│ Pencere │ Podyumla korelasyon     │ Boş satır    │
├─────────┼─────────────────────────┼──────────────┤
│ 3 yarış │ -0.466                  │ 244          │
├─────────┼─────────────────────────┼──────────────┤
│ 4 yarış │ -0.486                  │ 323          │
├─────────┼─────────────────────────┼──────────────┤
│ 5 yarış │ -0.493                  │ 402          │
├─────────┼─────────────────────────┼──────────────┤
│ 8 yarış │ -0.508                  │ 634          │
└─────────┴─────────────────────────┴──────────────┘

(Korelasyon negatif çünkü pozisyon küçüldükçe -yani pilot öne çıktıkça- podyuma girme ihtimali
artıyor. Mutlak değeri ne kadar büyükse ilişki o kadar güçlü.)

İlk bakışta 8 kazanıyor gibi duruyor: en güçlü korelasyon onda. Ama iki sebeple 3'ü seçtik.

Birincisi, pencere uzadıkça ölçtüğümüz şey değişiyor. 3 yarışlık ortalama "bu pilot şu sıralar
nasıl gidiyor" sorusunun cevabı — yani form. 8 yarışlık ortalama ise neredeyse "bu pilot genel
olarak ne kadar iyi" demeye başlıyor, yani forma değil sınıfa/seviyeye yaklaşıyor. Korelasyonun
yükselmesinin sebebi de tam bu: uzun ortalama gürültüyü siliyor ve pilotun sabit kalitesine
yakınsıyor. Oysa o bilgiyi zaten başka feature'lardan alacağız (sezon kümülatif puanı,
grid-bitiş farkı). Aynı şeyi iki kere ölçmenin modele faydası yok, sadece sütunlar birbirini
tekrar eder.

İkincisi, uzun pencere veri yiyor. 8'e çıkınca 634 satır boşalıyor; 3'te bu sayı 244. Elimizde
6.870 satır var ve kriterin alt sınırı 5.000 — her feature'da birkaç yüz satır kaybetmeyi göze
alırsak, yedi feature sonunda sınıra tehlikeli biçimde yaklaşırız.

Kısacası: kazanç küçük (0.466 → 0.508), bedeli ise hem veri kaybı hem de feature'ın anlamının
bulanması. 3 yarış "son form" fikrini en temiz anlatan pencere, onda kaldık.
'''



df[df['driver_id'] == 'alonso'].head(50)


#Oluşturduğumuz Rolling Form feature'u ne kadar etkili görelim

# Bosluk + dagilim + form araligina gore podyum orani (tablo ve grafik)
feature_summary(df, "rolling_form", plot=True)

# Dagilim bulutu + egilim dogrusu
feature_vs_position(df, "rolling_form")

# Tek pilotun kariyeri uzerinde feature ile gercek sonucun karsilastirmasi
driver_timeline(df, "alonso", ["rolling_form"])



# Not: correlation_matrix tek feature'la anlamsiz, en sondaki kontrol blogunda
# tum turetilmis sutunlar birden verilecek.






# %%
# 2. Sezon kumulatif puani
'''
Rolling form son 3 yarisa bakiyor, yani kisa vadeli bir sinyal. Ama bir pilotun o sezon ne kadar
guclu oldugunu 3 yaris anlatmaz - sezon boyunca biriktirdigi puan anlatir. 10. yarista 200 puani
olan pilot iyi bir arabada oturuyordur, 10 puani olan degil.

Ikinci feature'imiz bu: pilotun o sezon, o yarisa GIRERKEN sahip oldugu toplam puan.
Rolling form'dan farki ufku - biri "su sira nasil gidiyor", digeri "bu sezon nerede duruyor" der.

Bir seyi netlestirelim: karar defterinde points sutunu modele girdi olarak KULLANILMAYACAK
yaziyor. O kural o yarisin puani icin gecerli, cunku yaris bitmeden bilinmez. Burada
kullandigimiz onceki yarislarin puani - yaristan once elimizde olan bir bilgi. shift(1) tam
olarak bu ayrimi sagliyor.
'''

# Adim 1
# Nasil hesaplayacagiz?
'''
Adim 1 - Gruplama: driver_id VE season birlikte. Puanlar her sezon sifirlaniyor, 2023'un puani
2024'e tasinmaz. Tek basina driver_id'ye gruplarsak tum kariyer boyunca toplamaya devam eder.

Adim 2 - shift(1): rolling form'daki ayni sizinti onlemi.

Adim 3 - cumsum(): kaydirilmis puanlari bastan itibaren topla. 4. yarista ilk 3 yarisin puan
toplamini verir.

Sezonun ilk yarisinda sonuc NaN cikiyor (370 satir) ama onu 0 ile dolduruyoruz - cunku burada
"bilinmiyor" degil, herkes gercekten sifir puanla basliyor. Rolling form'daki bosluktan farki bu:
orada bilgi yoktu, burada bilgi var ve degeri sifir.

Not: siralama zaten onceki blokta yapildi (driver_id, season, round), tekrar siralamiyoruz.
'''

df['season_points'] = (
    df.groupby(['driver_id', 'season'])['points']
    .transform(lambda s: s.shift(1).cumsum())
    .fillna(0)
)

feature_summary(df, "season_points", bins=12, plot=True)
feature_vs_position(df, "season_points")




# %%
# 3. Takim son formu




# %%
# 4. Pist gecmisi




# %%
# 5. Takim arkadasi farki




# %%
# 6. Ariza orani




# %%
# 7. Grid-bitis farki




# %%
# Caylak pilot stratejisi (train'de hic gorulmemis pilot/takim + sezon basi gecmissizlik)




# %%
# Kolon sayisi kontrolu + kaydetme
