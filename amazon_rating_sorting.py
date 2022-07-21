# E-ticaretteki en önemli problemlerden bir tanesi ürünlere satış sonrası verilen puanların doğru şekilde hesaplanmasıdır.
# Bu problemin çözümü e-ticaret sitesi için daha fazla müşteri memnuniyeti sağlamak, satıcılar için ürünün öne çıkması
# ve satın alanlar için sorunsuz bir alışveriş deneyimi demektir. Bir diğer problem ise ürünlere verilen yorumların
# doğru bir şekilde sıralanması olarak karşımıza çıkmaktadır. Yanıltıcı yorumların öne çıkması ürünün satışını doğrudan
# etkileyeceğinden dolayı hem maddi kayıp hem de müşteri kaybına neden olacaktır. Bu 2 temel problemin çözümünde ,
# e-ticaret sitesi ve satıcılar satışlarını arttırırken müşteriler ise satın alma yolculuğunu sorunsuz olarak
# tamamlayacaktır.

# reviewerID: Kullanıcı ID’si
# asin : Ürün ID’si
# reviewerName:  Kullanıcı Adı
# helpful: Faydalı değerlendirme derecesi
# reviewText: Değerlendirme
# overall: Ürün rating’i
# summary: Değerlendirme özeti
# unixReviewTime: Değerlendirme zamanı
# reviewTime: Değerlendirme zamanı Raw
# day_diff: Değerlendirmeden itibaren geçen gün sayısı
# helpful_yes: Değerlendirmenin faydalı bulunma sayısı
# total_vote: Değerlendirmeye verilen oy sayısı

import pandas as pd
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
import math
import scipy.stats as st

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df_=pd.read_csv("/Users/sedeftaskin/Desktop/Data_Science/VBO/Homeworks/WEEK4/Amazon_review/amazon_review.csv")
df=df_.copy()
df.head()

# Proje Görevleri
# Görev 1: Average Rating’i güncel yorumlara göre hesaplayınız ve var olan average rating ile kıyaslayınız.
# Paylaşılan veri setinde kullanıcılar bir ürüne puanlar vermiş ve yorumlar yapmıştır. Bu görevde amacımız verilen puanları tarihe göre
# ağırlıklandırarak değerlendirmek. İlk ortalama puan ile elde edilecek tarihe göre ağırlıklı puanın karşılaştırılması gerekmektedir.

# Adım1: Ürünün ortalama puanını hesaplayınız.
avg_puan=df["overall"].mean()

# Adım 2: Tarihe göre ağırlıklı puan ortalamasını hesaplayınız.
# • reviewTime değişkenini tarih değişkeni olarak tanıtmanız
# • reviewTime'ın max değerini current_date olarak kabul etmeniz
# • her bir puan-yorum tarihi ile current_date'in farkını gün cinsinden ifade ederek yeni değişken oluşturmanız ve gün cinsinden ifade edilen
# değişkeni quantile fonksiyonu ile 4'e bölüp (3 çeyrek verilirse 4 parça çıkar) çeyrekliklerden gelen değerlere göre ağırlıklandırma yapmanız gerekir.
# Örneğin q1 = 12 ise ağırlıklandırırken 12 günden az süre önce yapılan yorumların ortalamasını alıp bunlara yüksek ağırlık vermek gibi.


df["reviewTime"] = pd.to_datetime(df["reviewTime"])
current_date= df["reviewTime"].max()
df["days"] = (current_date - df["reviewTime"]).dt.days
df["days"].quantile([0, 0.25, 0.5, 0.75])

# ağırlıklandırılmış skor
time_based_avg=df.loc[df["days"]<280, "overall"].mean() * 0.28 + df.loc[(df["days"]>280)& (df["days"]<430), "overall"].mean() * 0.26 + \
df.loc[(df["days"]>430)& (df["days"]<600), "overall"].mean()* 0.24 + df.loc[df["days"]>600, "overall"].mean() *0.22

time_based_avg > avg_puan

# Adım 3: Ağırlıklandırılmış puanlamada her bir zaman diliminin ortalamasını karşılaştırıp yorumlayınız.

df.loc[df["days"]<280, "overall"].mean()
df.loc[(df["days"]>280)& (df["days"]<430), "overall"].mean()
df.loc[(df["days"]>430)& (df["days"]<600), "overall"].mean()
df.loc[df["days"]>600, "overall"].mean()

# the closer the date the higher the mean values

# Görev 2: Ürün için ürün detay sayfasında görüntülenecek 20 review’i belirleyiniz.
# Adım 1: helpful_no değişkenini üretiniz.
# • total_vote bir yoruma verilen toplam up-down sayısıdır.
# • up, helpful demektir.
#  • Veri setinde helpful_no değişkeni yoktur, var olan değişkenler üzerinden üretilmesi gerekmektedir.
# • Toplam oy sayısından (total_vote) yararlı oy sayısı (helpful_yes) çıkarılarak yararlı bulunmayan oy sayılarını (helpful_no) bulunuz.
df["helpful_no"]=df["total_vote"]-df["helpful_yes"]

# Adım 2: score_pos_neg_diff, score_average_rating ve wilson_lower_bound skorlarını hesaplayıp veriye ekleyiniz.
# • score_pos_neg_diff, score_average_rating ve wilson_lower_bound skorlarını hesaplayabilmek için score_pos_neg_diff,
#  score_average_rating ve wilson_lower_bound fonksiyonlarını tanımlayınız.
# • score_pos_neg_diff'a göre skorlar oluşturunuz. Ardından; df içerisinde score_pos_neg_diff ismiyle kaydediniz.
# • score_average_rating'a göre skorlar oluşturunuz. Ardından; df içerisinde score_average_rating ismiyle kaydediniz.
# • wilson_lower_bound'a göre skorlar oluşturunuz. Ardından; df içerisinde wilson_lower_bound ismiyle kaydediniz.

def score_pos_neg_dif(up,down):
    return up-down

def score_average_rating(up,down):
    if up + down == 0:
        return 0
    return up / (up + down)


def wilson_lower_bound(up, down, confidence=0.95):
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)


# score_pos_neg_diff
df["score_pos_neg_diff"] = df.apply(lambda x: score_pos_neg_dif(x["helpful_yes"],
                                                                             x["helpful_no"]), axis=1)

# score_average_rating
df["score_average_rating"] = df.apply(lambda x: score_average_rating(x["helpful_yes"], x["helpful_no"]), axis=1)

# wilson_lower_bound
df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)


#Adım 3: 20 Yorumu belirleyiniz ve sonuçları Yorumlayınız.
#• wilson_lower_bound'a göre ilk 20 yorumu belirleyip sıralayanız.
#• Sonuçları yorumlayınız.

df.sort_values(by="wilson_lower_bound",ascending=False).head(20)