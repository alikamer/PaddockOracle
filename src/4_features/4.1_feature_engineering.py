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
pd.set_option('display.max_rows', None)


# Hedef degisken: podyuma girdi mi (ilk 3)
df["is_podium"] = (df["position"] <= 3).astype(int)
df.groupby('driver_id')['position'].transform(lambda s: s.shift(1).rolling(3).mean())

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
'''
Rolling form pilota bakiyor, bu takima. Bir pilotun sonucu buyuk olcude arabasinin
gucune bagli - takimin son 3 yaristaki ortalama bitis sirasi o gucun olcusu.

Neden ayri bir sutun gerekiyor: pilot degisir ama araba kalir. Kariyerinin basindaki
bir pilotun rolling_form'u bos, ama takiminin formu bellidir. Rolling form'un bos
oldugu 244 satirin 174'unde team_form dolu - bosluklari tam da orada kapatiyor.

Dikkat edilecek nokta: bu ikisinin birbiriyle korelasyonu 0.89, yani buyuk olcude
ayni seyi olcuyorlar. Yine de ikisini de tutuyoruz, gerekcesi yukaridaki bosluk isi.
'''

# Adim 1
# Once her takimin her yaristaki ortalama sirasini cikariyoruz (iki aracin ortalamasi),
# pencereyi ondan sonra uyguluyoruz. Dogrudan satir uzerinden yapamayiz: ayni yarista
# ayni takimdan iki satir var, "son 3 yaris" penceresi her yarisi iki kez sayardi.

team = (df.groupby(['team_unified', 'season', 'round'], as_index=False)['position'].mean()
           .sort_values(['team_unified', 'season', 'round']))

team['team_form'] = team.groupby('team_unified')['position'].transform(
    lambda s: s.shift(1).rolling(3).mean())

####################df = df.merge(team[['team_unified', 'season', 'round', 'team_form']],    ################## dikkat!!
              on=['team_unified', 'season', 'round'], how='left')

df = df.sort_values(['driver_id', 'season', 'round']).reset_index(drop=True)  # merge sirayi bozdu
df.head()
feature_summary(df, "team_form", plot=True)
feature_vs_position(df, "team_form")


# %%
# 4. Pist gecmisi
'''
Bazi pilotlar belli pistlerde istikrarli sekilde iyi. Monaco gibi teknik pistlerde
pilotun kendisi, Monza gibi guc pistlerinde araba one cikar - yani pist pilot-araba
eslesmesini degistiriyor.

Model bunu nasıl kullanacak? Bir pilot genel formda kötü olsa bile,
o hafta yarış Monaco'daysa ve o pilotun Monaco geçmişi güçlüyse, model bunu ayrı bir sinyal olarak görüp podyum ihtimalini yukarı çekebilir.
rolling_form'un veremediği, pist-spesifik bir bilgi.

Bu sutun: pilotun bu piste daha once geldiginde aldigi ortalama sonuc. Burada pencere
kullanmiyoruz, expanding (bastan itibaren tumu) aliyoruz - pilot ayni piste yilda bir
kez geliyor, 3'luk bir pencere zaten neredeyse tum gecmisini kapsardi.

Bosluk orani yuksek olacak (%27): her pilotun her piste ilk gelisi bos kaliyor.
Caylak blogunda dolduruyoruz.

Not: siralama driver_id/season/round oldugu icin, pilot+pist alt gruplarindaki
satirlar da kendiliginden kronolojik sirada. Tekrar siralamaya gerek yok.
'''

df['circuit_history'] = df.groupby(['driver_id', 'circuit_id'])['position'].transform(
    lambda s: s.shift(1).expanding().mean())
df.head(200)
feature_summary(df, "circuit_history", plot=True)
feature_vs_position(df, "circuit_history")

#bütün pistlere bakmamak gerekebilir bu değişkende
'''
 pist  satir  korelasyon
        losail     47   -0.483451
     rodriguez    153   -0.442186
         buddh     35   -0.441735
         sochi    117   -0.397160
         imola     66   -0.370951
   silverstone    277   -0.368102
        jeddah     66   -0.365479
      shanghai    185   -0.357303
      americas    201   -0.340118
     zandvoort     68   -0.338208
        suzuka    225   -0.336678
         vegas     34   -0.335698
    interlagos    236   -0.330512
 red_bull_ring    227   -0.324242
    yas_marina    260   -0.322696
        monaco    239   -0.320065
        sepang    122   -0.318734
        ricard     49   -0.316446
    marina_bay    222   -0.309389
   hungaroring    263   -0.289688
    villeneuve    223   -0.278520
       bahrain    255   -0.276288
         monza    261   -0.275224
     catalunya    260   -0.254872
   albert_park    213   -0.224844
          baku    134   -0.176259
       yeongam     54   -0.172346
         miami     50   -0.143929
hockenheimring     75   -0.132419
      valencia     36   -0.073167
      istanbul     37   -0.061101
           spa    262   -0.03596
'''

#Pist bazlı analiz
def circuit_history_strength(dataframe, col="circuit_history", min_rows=30):
    """
    Her pist icin ayri ayri, col (gecmis ortalamasi) ile gercek pozisyon arasindaki
    korelasyonu hesaplar, tek bir yatay bar grafikte guclu -> zayif siralar.

    Amac: feature_vs_position'daki tek cizgi TUM pistleri karistiriyordu.
    Burada her pist kendi cubugunu aliyor - cubuk sifira ne kadar yakinsa
    o pistte gecmis o kadar az isaret ediyor demek.
    """
    subset = dataframe.dropna(subset=[col])

    result = (
        subset.groupby("circuit_id")
        .filter(lambda g: len(g) >= min_rows)
        .groupby("circuit_id")
        .apply(lambda g: g[col].corr(g["position"]))
        .sort_values()
    )

    plt.figure(figsize=(8, 10))
    result.plot(kind="barh", color="steelblue")
    plt.xlabel(f"{col} ile gercek pozisyon korelasyonu")
    plt.title(f"Pist basina {col} ne kadar isaret ediyor?")
    plt.axvline(0, color="black", linewidth=0.8)
    plt.tight_layout()
    plt.show()
circuit_history_strength(df, "circuit_history")




# %%
# 5. Takim arkadasi farki
'''
Ayni yarista, ayni arabayi kullanan iki pilot var. Aralarindaki fark arabadan
gelemez, pilottan gelir. Bu yuzden bu sutun elimizdeki en temiz "saf pilot yetenegi"
olcusu.

Hesap: pilotun gecmis yarislarda takim arkadasindan kac sira onde bitirdiginin
ortalamasi. Pozitif deger pilotun ustun oldugunu soyluyor.

Korelasyonu dusuk cikacak (0.14) ve bu beklenen bir sey - podyum buyuk olcude araba
isi, takim arkadasini yenmek tek basina podyuma yetmiyor. Yine de degerli, cunku
araba gucunden bagimsiz tek sinyalimiz bu.
'''

# Adim 1
# Takim arkadasinin sirasini ayri bir tablo kurmadan buluyoruz: ayni yarista ayni
# takimdan iki satir varsa, ortalamanin iki kati eksi kendi sirasi otekinin sirasidir.
# Tek arac cikaran takimlarda (veride 44 yaris) takim arkadasi yok, orasi bos kalir.

car_count = df.groupby(['team_unified', 'season', 'round'])['position'].transform('size')
team_avg = df.groupby(['team_unified', 'season', 'round'])['position'].transform('mean')

diff = (2 * team_avg - df['position']) - df['position']
df['_diff'] = diff.where(car_count == 2)

df['teammate_delta'] = df.groupby('driver_id')['_diff'].transform(
    lambda s: s.shift(1).expanding().mean())

feature_summary(df, "teammate_delta", plot=True,bins=12)
feature_vs_position(df, "teammate_delta")
'''
Hamilton  → pozisyon 3
Russell   → pozisyon 4

Ne yapmaya çalışıyoruz? Basit: her pilotun satırına, "takım arkadaşı kaçıncı oldu" bilgisini yazmak. Ama bunun için ayrı bir tablo kurup eşleştirmek yerine, ortaokul matematiğiyle bir kısayol kullanıyoruz.

Adım 1 — grup büyüklüğü ve ortalama:
Bu iki satır aynı takım+sezon+round grubunda. arac_sayisi = 2 (grupta 2 kişi var). takim_ort = (3+4)/2 = 3.5.

Adım 2 — kısayol: İki sayının ortalamasını biliyorsan ve birini biliyorsan, diğerini bulabilirsin:

$$\text{diğer sayı} = 2 \times \text{ortalama} - \text{bildiğin sayı}$$

Hamilton'ın satırında "bildiğin sayı" kendi pozisyonu (3). Takım arkadaşının (Russell'ın) pozisyonunu bul:

2 × 3.5 − 3 = 4   ← evet, Russell gerçekten 4. oldu

Russell'ın satırında da aynı formül, bu sefer "bildiğin sayı" = 4:

2 × 3.5 − 4 = 3   ← evet, Hamilton gerçekten 3. oldu

Adım 3 — fark: Şimdi her satırda hem kendi pozisyon hem takım arkadaşının pozisyonu var, farkını alıyoruz:

Hamilton satırı: fark = takım_arkadaşı(4) − kendi(3) = +1   → arkadaşından 1 sıra önde
Russell satırı:  fark = takım_arkadaşı(3) − kendi(4) = −1   → arkadaşından 1 sıra geride

Pozitif = pilot üstün, negatif = pilot geride. Mantıklı: Hamilton önde bitirdi, +1 aldı.

Adım 4 — teammate_delta: Bu fark tek bir yarışın sonucu, ama biz pilotun genel eğilimini istiyoruz. O yüzden her pilot için, o ana kadarki tüm geçmiş yarışlarının fark değerlerinin ortalamasını alıyoruz (yine shift(1) var — bu yarışın kendi farkı sayılmıyor, sadece geçmiş).

Yani sonuçta teammate_delta şunu söylüyor: "bu pilot, kariyeri boyunca takım arkadaşlarını ortalama kaç sıra farkla geçiyor/kaybediyor."

Tek araçlı takımlarda (Williams, Albon örneği gibi) takım arkadaşı yok, arac_sayisi 1 çıkıyor, o satırlar NaN bırakılıyor.'''

#|teammate_delta| <= 0.5 : %27.4 satır
#|teammate_delta| <= 1   : %51.1 satır   ← yarısı burada
#|teammate_delta| <= 2   : %81.3 satır
#|teammate_delta| <= 3   : %89.9 satır




# %%
# 6. Ariza orani
'''
Yarisi bitiremeyen pilot podyuma cikamaz. Bu sutun pilotun gecmiste ne siklikta
bitiremedigini tutuyor.

"Bitirdi" saydiklarimiz: status degeri "Finished", "+N Lap" (tur geriden bitirenler)
ve "Lapped". Geri kalan 70'ten fazla deger (Engine, Collision, Gearbox, Accident,
Retired...) bitirememe sayiliyor. Veride DNF orani %17.

Bu sutun sifira yigilmis, yani carpik. O yuzden bin sayisini 8'e cikariyor,
feature_vs_position'i da atliyoruz - carpik sutunlarda o grafik yaniltiyor.
'''

finished = df['status'].eq('Finished') | df['status'].str.startswith('+') | df['status'].eq('Lapped')
df['_dnf'] = (~finished).astype(int)

df['dnf_rate'] = df.groupby('driver_id')['_dnf'].transform(
    lambda s: s.shift(1).expanding().mean())
feature_summary(df, "dnf_rate", plot=True)


# %%
# 7. Grid-bitis farki
'''
Pilot basladigi yerden kac sira kazaniyor? grid - position pozitifse ileri tirmanmis,
negatifse gerilemis. Gecmis ortalamasi pilotun yaris ici verimini gosterir.

Onemli uyari: bu olcum basladigi yerle sinirli. Pole'dan baslayan en fazla 0 sira
kazanabilir, 18. baslayan rahatca 8 kazanir. Bu yuzden podyumla korelasyonu NEGATIF
cikiyor (-0.23) - cok sira kazananlar aslinda gerilerden baslayanlar.

Sutunu yine de tutuyoruz: model grid'i de goruyor, dolayisiyla "15.den 3 sira
kazanmis" ile "5.den 3 sira kazanmis" arasindaki farki kendisi kurabiliyor.
'''

df['_gain'] = df['grid'] - df['position']

df['grid_gain'] = df.groupby('driver_id')['_gain'].transform(
    lambda s: s.shift(1).expanding().mean())

feature_summary(df, "grid_gain", plot=True)
_bins = pd.qcut(df['grid_gain'].dropna(), 5, duplicates='drop', precision=1)
print(df.dropna(subset=['grid_gain']).groupby(_bins, observed=True)['grid'].mean())

feature_vs_position(df, "grid_gain")


'''
"grid_gain, bir pilotun geçmişte ortalama kaç sıra kazandığını ölçüyor. Podyumla ilişkisi ters çıktı çünkü çok sıra kazananlar aslında geriden başlayan pilotlar — 8 sıra kazansalar bile 18.'den kalkmışlarsa hâlâ podyuma uzaklar. Bu yüzden sütunu tek başına değil, grid ile birlikte kullanıyoruz; model ikisini birleştirip pilotun hem nereden başladığını hem yarış içi performansını ayrı ayrı değerlendirebiliyor."

'''

# %%
# Caylak pilot stratejisi (gecmissiz satirlarin doldurulmasi)
'''
Turetilen sutunlarin hepsi gecmise bakiyor, dolayisiyla gecmisi olmayan satirlarda
bos kaliyorlar. En buyuk bosluk pist gecmisinde (%27), en kucugu sezon puaninda (sifir).

Iki adimda cozuyoruz:

1) career_races sutunu - pilotun bu yaristan onceki toplam yaris sayisi, debutte 0.
   Bu bir doldurma degil, gercek bir feature: modele "bu satirin gecmisi ne kadar
   saglam" bilgisini veriyor. Boylece doldurulmus degerlere ne kadar guvenecegini
   model kendi ogreniyor.

2) Kalan bosluklari sutunun medyani ile dolduruyoruz. Medyan ortalamadan guvenli,
   uctaki birkac deger onu cekemiyor.

Not: medyani simdilik tum veriden hesapliyoruz. Siki konusursak bu kucuk bir
sizintidir - dogrusu sadece train setinin medyanini kullanmak. 5_models'te kronolojik
bolme yapilinca bu hesap oraya tasinacak.
'''

df['career_races'] = df.groupby('driver_id').cumcount()

derived_cols = ['rolling_form', 'season_points', 'team_form', 'circuit_history',
             'teammate_delta', 'dnf_rate', 'grid_gain', 'career_races']

for col in derived_cols:
    df[col] = df[col].fillna(df[col].median())

print("Doldurma sonrasi kalan bosluk:")
print(df[derived_cols].isnull().sum().to_string())

feature_summary(df, "career_races", bins=8, plot=True)


# %%
# Kolon sayisi kontrolu + kaydetme
'''
Son kontrol. Uc soru soruyoruz: yardimci sutunlar temizlendi mi, sizinti sutunlari
modele girmiyor mu, kriterin istedigi feature sayisina ulastik mi (min 10, ideal 15-30).
'''

df = df.drop(columns=[col for col in df.columns if col.startswith('_')])

# Yaristan SONRA olusan sutunlar. Modele girdi olarak asla verilmeyecek
# (karar defteri, 02.09.2026). position sadece hedef degiskeni turetmek icin kullanildi.
leakage_cols = ['position', 'position_text', 'points', 'laps', 'status']

# Yaristan ONCE bilinen her sey modele girebilir.
numeric_cols = ['grid', 'rolling_form', 'season_points', 'team_form', 'circuit_history',
                    'teammate_delta', 'dnf_rate', 'grid_gain', 'career_races', 'season', 'round']
# driver_id bilerek yok: karar defterinde kimlik sutunlarina one-hot yapilmayacak
# yaziyor (83 pilot = 83 sutun). Pilotun kimligi yerine davranisi duruyor:
# rolling_form, career_races, teammate_delta, grid_gain, dnf_rate.
categorical_cols = ['team_unified', 'circuit_id']
model_cols = numeric_cols + categorical_cols

assert df[model_cols].isnull().sum().sum() == 0, "model sutunlarinda bos hucre var"
assert not set(model_cols) & set(leakage_cols), "sizinti sutunu model listesine karismis"
assert len(model_cols) >= 10, "kriterin alt siniri 10 feature"
assert df['is_podium'].isin([0, 1]).all(), "hedef 0/1 disinda deger tasiyor"

print(f"{len(df)} satir | {len(model_cols)} model sutunu | {len(df.columns)} toplam sutun")
print(f"podyum orani: %{100 * df['is_podium'].mean():.1f}")
print(f"sayisal {len(numeric_cols)}, kategorik {len(categorical_cols)}")

# Turetilen sutunlar birbirini tekrar ediyor mu? (rolling_form <-> team_form'u burada gor)
correlation_matrix(df, numeric_cols + ['is_podium'])

output_path = "data/data_processed/results_2010_2025_features.csv"
df.to_csv(output_path, index=False)
print(f"Kaydedildi: {output_path}")
