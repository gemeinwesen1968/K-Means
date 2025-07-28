#import "@preview/basic-report:0.3.0": *

#show: it => basic-report(
  doc-title: "Kümeleme Çalışması",
  author: "Göktuğ Türkmen",
  compact-mode: false,
  it
)

#set text(size: 10pt, lang: "tr")

#set par(
  first-line-indent: 1em,
  spacing: 0.65em,
  justify: true,
)

= Çalışmanın Amacı

Bu çalışmanın amacı, müşteri verilerini kümeleme yöntemleriyle analiz ederek benzer özelliklere sahip müşteri gruplarını ortaya çıkarmaktır. Farklı türde kümeleme algoritmaları araştırılmış ve özellikle KMeans ve KMeans++ algoritmaları uygulanmıştır. Bu algoritmalar, hazır kütüphaneler kullanılmadan Python ile geliştirilmiştir.#footnote[Algoritmalar geliştirilirken yalnızca temel vektör ve matris işlemleri için NumPy kütüphanesinden yararlanılmıştır.]

= Kümeleme

Kümeleme, bir gözetimsiz öğrenme (unsupervised learning) yöntemidir. Bir veri setindeki verilerin birbirlerine olan benzerliklerine#footnote[Benzerliğin kaynağı algoritmadan algoritmaya değişmektedir. Veri setinin yapısı da seçilen benzerlik kriteri üzerinde etkendir.] göre belirli sayıdaki kümelere ayrılması işlemidir. Varılacak kümeler hakkında verili bir ön bilgi olmadığı için gözetimsiz öğrenme kategorisindedir. Kümeleme, özellikle etiketlenmemiş veriler üzerinde gizli yapıları keşfetmek ve veri segmentasyonu yapmak için sıkça kullanılır.

== KMeans Kümeleme

KMeans en sık kullanılan kümeleme algoritmalarındandır. Ağırlık merkezi temellidir ve benzerlik kriteri olarak küme ağırlık merkezine olan mesafeyi alır#footnote[Mesafe kriteri olarak genellikle Öklid uzaklığı alınır. Farklı uzaklık çeşitlerini de kullanmak mümkündür.]. Elde edilecek küme sayısı algoritmaya önceden verilmelidir. En optimal küme sayısını elde etmek için Elbow method ve benzeri yöntemler kullanılmaktadır. Algoritmanın aşamaları:
+ Küme sayısı `k` belirlenir.
+ Başlangıçta `k` adet ağırlık merkezi rastgele seçilir.
+ Her veri noktası, kendisine en yakın merkeze atanır.
+ Her küme için yeni merkez, o kümeye ait veri noktalarının ortalaması alınarak güncellenir.
+ 3 ve 4. adımlar, merkezler değişmeyene kadar tekrar edilir.

Aşağıda Python ile yazılmış basit bir KMeans algoritması örneği verilmiştir.

#linebreak()
```python
def kmeans(X, k, kmeans_pp=False, max_iter=100, tol=1e-4, elbow=False):
  if kmeans_pp:
    centroids = init_centroid_kmeanspp(X, k)
  else
    centroids = init_centroid(X, k)
  clusters = []
  for _ in range(max_iter):
    clusters = assign_clusters(X, centroids)
    new_centroids = update_centroids(X, clusters, k)
    if np.all(np.linalg.norm(new_centroids - centroids, axis=1) < tol):
      break
    centroids = new_centroids
  wcss = 0
  if elbow:
    for i in range(k):
      points = X[clusters == i]
      wcss += np.sum((points - centroids[i]) ** 2)
  return clusters, centroids, wcss
```

== KMeans++ Kümeleme

KMeans++ algoritması, KMeans algoritmasının geliştirilmiş bir versiyonudur. İyileştirme, başlangıç merkezlerinin daha dengeli ve stratejik şekilde seçilmesine dayanır. KMeans algoritmasında başlangıç merkezleri tamamen rastgele seçildiği için sonuçlar kararsız olabilir ve lokal minimumlara sapma riski yüksektir. KMeans++ ise bu problemi azaltmak için daha sofistike bir başlangıç stratejisi uygular:
+ İlk merkez rastgele seçilir.
+ Sonraki her nokta, mevcut merkezlere olan uzaklıklarının karesi oranında bir olasılıkla merkez adayı olarak seçilir.
+ Bu işlem, `k` adet merkez seçilene kadar tekrarlanır.
Aşağıda KMeans++ için kullanılan başlangıç merkezi seçme fonksiyonu yer almaktadır:

#linebreak()
```python
def init_centroid_kmeanspp(X, k):
  centroids = []
  centroids.append(X[np.random.choice(len(X))])
  for _ in range(1, k):
    dist_sq = np.array([min(np.sum((x - c)**2) for c in centroids) for x in X])
    probs = dist_sq / dist_sq.sum()
    cumulative_probs = probs.cumsum()
    r = np.random.rand()
    for i, p in enumerate(cumulative_probs):
      if r < p:
        centroids.append(X[i])
        break
  return np.array(centroids)
```

= Veri Seti

Bu çalışmada kullanılan veri seti#footnote[Veri setinin okunması ve işlenmesi için Pandas kütüphanesi kullanılmıştır.], müşterilere ait demografik bilgiler, satın alma geçmişi ve kampanya etkileşimlerini içermektedir. Veri setindeki özellikler aşağıda verilmiştir:

- *ID* — Müşteriye ait benzersiz tanımlayıcı (kategorik)
- *Year_Birth* — Müşterinin doğum yılı (sayısal)
- *Education* — Eğitim durumu (kategorik)
- *Marital_Status* — Medeni durumu (kategorik)
- *Income* — Yıllık hane geliri (sayısal)
- *Kidhome* — Evde yaşayan çocuk sayısı (sayısal)
- *Teenhome* — Evde yaşayan genç sayısı (sayısal)
- *Dt_Customer* — Müşterinin sisteme katıldığı tarih (tarih/sayısal)
- *Recency* — Son alışverişin üzerinden geçen gün sayısı (sayısal)
- *Complain* — Son 2 yılda şikayet bildirip bildirmediği (1 = evet, 0 = hayır) (kategorik)
- *MntXXX* — Son 2 yılda XXX ürün grubuna harcanan miktar (sayısal)  
  (Wines, Fruits, MeatProducts, FishProducts, SweetProducts, GoldProds)
- *NumDealsPurchases* — İndirimli alışveriş sayısı (sayısal)
- *AcceptedCmpX* — X numaralı kampanya teklifi kabul edildi mi (1 = evet, 0 = hayır)  
  (Cmp1, Cmp2, Cmp3, Cmp4, Cmp5) (kategorik)
- *Response* — Son kampanya teklifini kabul edip etmediği (1 = evet, 0 = hayır) (kategorik)

= Veri Ön Hazırlığı

Veri ön hazırlığı aşamasında, eksik ve aykırı değerlerin temizlenmesi, kategorik verilerin dönüştürülmesi ve değişkenlerin ölçeklendirilmesi işlemleri uygulanmıştır. Kümelenebilir özelliklerin daha anlamlı hale getirilmesi için ön hazırlık gereklidir. 

- Eksik veriler (`NaN`) veri setinden çıkarılmıştır.
- Değişkenliği olmayan sabit sütunlar (`Z_CostContact`, `Z_Revenue`) veri setinden kaldırılmıştır.

- `Age`: Kişinin yaşı, doğum yılına göre hesaplanmıştır (`2025 - Year_Birth`).
- `Dt_Customer`: En güncel kayıt tarihine göre müşterinin kayıt süresi gün cinsinden hesaplanmıştır.
- `Children`: Evdeki çocuk sayısı (`Teenhome + Kidhome`) olarak tanımlanmıştır.
- `Parent`: Çocuğu olan bireyler için `1`, olmayanlar için `0` değeri atanmıştır.
- `Marital_Status`: `YOLO` ve `Absurd` gibi anlam taşımayan kategoriler çıkarılmıştır.
- `Marital_Status`: `Married` ve `Together` → `Partner`, diğerleri (`Widow`, `Divorced`, `Single`) → `Alone` olarak gruplanmıştır.
- `Education`: Eğitim seviyesi `Low` (Graduation, 2n Cycle) ve `High` (PhD, Master) olmak üzere iki gruba indirgenmiştir.
- `Tot_Mnt`: Toplam harcama, 6 kategori (`MntWines`, `MntFruits`, `MntMeatProducts`, `MntFishProducts`, `MntSweetProducts`, `MntGoldProds`) üzerinden toplanmıştır.
- `Tot_Purchase`: Toplam alışveriş sayısı, farklı kanallar üzerinden toplanmıştır.
- `Tot_Accepted`: Tüm kampanyalara verilen olumlu cevapların toplamı.
- `Age` ≥ 90 ve `Income` ≥ 600000 olan gözlemler çıkarılmıştır.
- Harcama sütunlarında aykırı değerler, genişletilmiş IQR yöntemiyle (çarpan = 4.0) temizlenmiştir.
- `Education` ve `Marital_Status` değişkenleri one-hot encoding ile sayısal forma çevrilmiştir.
- `ID`, `Year_Birth` ve kampanya sütunları dahil, modelde kullanılmayan sütunlar veri setinden çıkarılmıştır.

#linebreak()
Kümeleme algoritmalarında kullanılmak üzere aşağıdaki özellikler seçilmiştir:

- `Age`
- `Income`
- `Tot_Mnt`
- `Tot_Purchase`
- `Dt_Customer`
- `Education_High`
- `Parent`
- `Marital_Status_Partner`
- `Tot_Accepted`

= Veri Üzerinde Kümeleme ve Görselleştirme

== Elbow Yöntemi ile Küme Sayısı Belirleme

Kümeleme öncesi Elbow yöntemi kullanılarak küme sayısının 4 olmasına karar verilmiştir. Elbow yöntemi, farklı küme sayılarına göre K-Means algoritmasını çalıştırarak her küme sayısı için toplam hata (inertia) değerini hesaplar. Bu değer, verilerin kümelere ne kadar iyi ayrıldığını gösterir. Genellikle küme sayısı arttıkça hata azalır; fakat belirli bir noktadan sonra bu azalma önemsiz hale gelir. Bu kırılma noktası, grafikte bir “dirsek” (elbow) gibi görünür ve en uygun küme sayısı olarak kabul edilir.

#linebreak()
Bu çalışmada:

- Küme sayısı 1'den 10'a kadar denenmiştir.
- Inertia değerleri çizdirilmiştir.
- Eğrinin kırılma yaptığı nokta k = 4 olarak belirlenmiştir.

== Kümeleme Sonucunu Görselleştirme

Kümeleme işlemi, seçilen öznitelikler (`Age`, `Income`, `Tot_Mnt`, `Tot_Purchase`, vb.) üzerinden gerçekleştirilmiştir. Ancak elde edilen kümeler, çok boyutlu bir uzayda yer aldığından dolayı doğrudan görselleştirilemez. Bu sebeple boyut indirgeme tekniklerinden faydalanılmıştır.


=== PCA ile Görselleştirme

PCA (Principal Component Analysis), yüksek boyutlu verileri en fazla varyansı koruyarak daha düşük boyutlu bir uzaya indirger. Bu çalışmada, kümeleri 3 boyutlu uzayda gözlemleyebilmek için uygulanmıştır.#footnote[PCA için sklearn kütüphanesi kullanılmıştır.] 

#linebreak()
#figure(
  image("plots/cluster/pca_3d_clusters.png", width: 40%),
  caption: [
    Kümelerin PCA ile 3 boyuta indirgenerek görselleştirilmesi. 
  ],
) <PCA>

= Kümeler ve Özellikleri

Kümeleme sonrası her bir kümenin kendine has karakteristikleri bulunmaktadır. Bu karakteristikleri inceleyebilmek için önemli özniteliklerin (yaş, gelir, toplam harcama vb.) kümelere göre dağılımları incelenmiştir.

Aşağıdaki görsellerde, farklı özellikler için küme bazında dağılım grafiklerine yer verilmiştir. Bu sayede kümelerin hangi özelliklerde farklılaştığı gözlemlenebilir.

#linebreak()
#figure(
  image("plots/cluster/cluster_analysis_plots.png", width: 60%),
  caption: [
    Kümelerin toplam harcamanın diğer sayısal değerlere göre grafiğinde oluşturduğu dağılım. 
  ]
) <cluster_analysis>

#linebreak()
#figure(
  image("plots/cluster/cluster_totmnt_Age.png", width: 50%),
  caption: [Toplam harcama ve yaşın kümelere dağılımı.]
) <totmnt_age>

#linebreak()
#figure(
  image("plots/cluster/cluster_totmnt_Children.png", width: 50%),
  caption: [Toplam harcamanın ve çocuk sayısının kümelere dağılımı.]
) <totmnt_children>

#linebreak()
#figure(
  image("plots/cluster/cluster_totmnt_Dt_Customer.png", width: 50%),
  caption: [Toplam harcama ve günün kümelere dağılımı.]
) <totmnt_dt>

#linebreak()
#figure(
  image("plots/cluster/cluster_totmnt_Education_Low.png", width: 50%),
  caption: [Toplam harcama ve eğitim seviyesinin kümelere dağılımı.]
) <totmnt_edu>

#linebreak()
#figure(
  image("plots/cluster/cluster_totmnt_Marital_Status_Partner.png", width: 50%),
  caption: [Toplam harcama ve medeni durumun kümelere dağılımı.]
) <totmnt_partner>

#linebreak()
#figure(
  image("plots/cluster/cluster_totmnt_Parent.png", width: 50%),
  caption: [Toplam harcama ve ebeveynlik durumunun kümelere dağılımı.]
) <totmnt_parent>

#linebreak()
#figure(
  image("plots/cluster/cluster_totmnt_Tot_Purchase.png", width: 50%),
  caption: [Toplam harcama ve kabul edilen pazarlamanın kümelere dağılımı.]
) <totmnt_purchase>

#linebreak()
#figure(
  image("plots/cluster/cluster_meat.png", width: 60%),
  caption: [Kümelerin satın alınan et miktarına göre dağılımı.]
) <cluster_meat>

#linebreak()
#figure(
  image("plots/cluster/cluster_wines.png", width: 60%),
  caption: [Kümelerin satın alınan şarap miktarına göre dağılımı.]
) <cluster_wine>

#linebreak()
== Küme 0
- Genel olarak en yüksek harcama miktarına sahip. (#ref(<cluster_analysis>))
- Genel olarak ebeveyn değiller. (#ref(<totmnt_children>) ve #ref(<totmnt_parent>))
- Et ve şarap gibi daha pahalı ürünleri en çok alan küme. (#ref(<cluster_meat>) ve #ref(<cluster_wine>))
- Kampanyaların en çok kabul eden küme. (#ref(<totmnt_purchase>) ve #ref(<cluster_analysis>))

== Küme 1
- En az harcama miktarına sahip. (#ref(<cluster_analysis>))
- Kampanyaları en az kabul eden küme. (#ref(<totmnt_purchase>))
- Büyük bir kısmı evli ve çocuklu. (#ref(<totmnt_partner>) ve #ref(<totmnt_parent>))
- Önemli bir kısmı neredeyse hiç et ve şarap gibi ürünleri almıyor. (#ref(<cluster_meat>) ve #ref(<cluster_wine>))
- Çoğunlukla eğitim seviyesi görece daha düşük. (#ref(<totmnt_edu>))

== Küme 2
- Küme 0'dan sonra en fazla harcama miktarına sahip. (#ref(<cluster_analysis>))
- Et ürünlerini pek tercih etmese de şarapta Küme 0'a yakınlar. (#ref(<cluster_meat>) ve #ref(<cluster_wine>))
- Genel olarak evliler ve ebeveynler. (#ref(<totmnt_parent>), #ref(<totmnt_partner>))
- Çocuk sayısı Küme 1 ve 3'e karşı genel olarak daha az. (#ref(<totmnt_children>))
- Kampanyalara karşı Küme 0'a göre daha temkinliler. (#ref(<cluster_analysis>))

== Küme 3
- Şarap ürününde Küme 1'e göre daha fazla satın alım yapmaktadır. (#ref(<cluster_wine>))
- Toplam harcama olarak genelde düşük olsa da Küme 1'in üstüne (Küme 2'nin alt harcama seviyesine) çıkabilmektedir. (#ref(<cluster_analysis>))

= Sonuç

Bu çalışmada, bir müşteri veri seti üzerinde kümeleme algoritmaları uygulanarak müşteri segmentasyonu gerçekleştirilmiştir. Öncelikle verideki aykırı değerler ve eksik veriler temizlenmiş, ardından sayısal öznitelikler ölçeklendirilerek analiz için uygun hale getirilmiştir. KMeans algoritması kullanılarak yapılan kümeleme sonucunda, Elbow yöntemiyle en uygun küme sayısı 4 olarak belirlenmiştir.

Kümeleme sonrası her bir kümenin özniteliklere göre farklılık gösterdiği gözlemlenmiştir. Örneğin:

- Küme 0, yüksek gelir ve yüksek toplam harcama ile dikkat çekerken, genellikle çocuk sahibi olmayan bireylerden oluşmaktadır.

- Küme 1, düşük harcama ve etkileşim düzeyi ile daha pasif müşteri grubunu temsil etmektedir.

- Küme 2, kampanya kabul oranı yüksek ve ebeveyn olma olasılığı fazla olan bireyleri içermektedir.

- Küme 3 ise genellikle orta düzeyde harcama yapan, eğitim seviyesi görece yüksek bireylerden oluşmaktadır.

= Kaynaklar

+ Glenn Fung, *A Comprehensive Overview of Basic Clustering Algorithms*, 22 Haziran 2001.
+ Laurenz Wiskott, *Lecture Notes on Principal Component Analysis*, ilk sürüm: 11 Mart 2004, son güncelleme: 21 Şubat 2013.
+ Imakash3011. *Customer Personality Analysis Dataset*. Kaggle. https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis
