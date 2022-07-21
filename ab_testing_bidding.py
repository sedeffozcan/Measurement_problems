import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# !pip install statsmodels
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, \
    pearsonr, spearmanr, kendalltau, f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

# İş Problemi
# Facebook kısa süre önce mevcut "maximum bidding" adı verilen teklif verme türüne alternatif olarak yeni bir
# teklif türü olan "average bidding"’i tanıttı.
# Müşterilerimizden biri olan bombabomba.com, bu yeni özelliği test etmeye karar verdi ve average bidding'in
# maximum bidding'den daha fazla dönüşüm getirip getirmediğini anlamak için bir A/B testi yapmak istiyor.
# A/B testi 1 aydır devam ediyor ve bombabomba.com şimdi sizden bu A/B testinin sonuçlarını analiz etmenizi
# bekliyor. Bombabomba.com için nihai başarı ölçütü Purchase'dır. Bu nedenle, istatistiksel testler için
# Purchase metriğine odaklanılmalıdır.

# Veri Seti Hikayesi
# Bir firmanın web site bilgilerini içeren bu veri setinde kullanıcıların gördükleri ve tıkladıkları reklam
# sayıları gibi bilgilerin yanı sıra buradan gelen kazanç bilgileri yer almaktadır. Kontrol ve Test grubu olmak
# üzere iki ayrı veri seti vardır. Bu veri setleri ab_testing.xlsx excel’inin ayrı sayfalarında yer almaktadır.
# Kontrol grubuna Maximum Bidding, test grubuna Average Bidding uygulanmıştır.

# Impression: Reklam görüntüleme sayısı
# Click: Görüntülenen reklama tıklama sayısı
# Purchase: Tıklanan reklamlar sonrası satın alınan ürün sayısı
# Earning: Satın alınan ürünler sonrası elde edilen kazanç

# Proje Görevleri

# Adım 1: ab_testing_data.xlsx adlı kontrol ve test grubu verilerinden oluşan veri setini okutunuz.
# Kontrol ve test grubu verilerini ayrı değişkenlere atayınız.

df1=pd.read_excel("/Users/sedeftaskin/Desktop/Data_Science/VBO/Homeworks/WEEK4/AB_testing/ab_testing.xlsx",sheet_name="Control Group")
df2=pd.read_excel("/Users/sedeftaskin/Desktop/Data_Science/VBO/Homeworks/WEEK4/AB_testing/ab_testing.xlsx",sheet_name="Test Group")
df_control=df1.copy()
df_test=df2.copy()
df_test=df_test[["Impression","Click","Purchase","Earning"]]
df_control=df_control[["Impression","Click","Purchase","Earning"]]
# Adım 2: Kontrol ve test grubu verilerini analiz ediniz.
df_test["durum"]="test"
df_control["durum"]="control"
df_control.describe().T
df_test.describe().T


# Adım 3: Analiz işleminden sonra concat metodunu kullanarak kontrol ve test grubu verilerini birleştiriniz.

df=pd.concat([df_control,df_test],ignore_index=True)
df.head()

# Görev 2: A/B Testinin Hipotezinin Tanımlanması
# Adım 1: Hipotezi tanımlayınız.
# H0 : M1 = M2
# H1 : M1!= M2


# Adım 2: Kontrol ve test grubu için purchase (kazanç) ortalamalarını analiz ediniz.
df_control["Purchase"].mean()
df_test["Purchase"].mean()

# Görev 3: Hipotez Testinin Gerçekleştirilmesi

# Adım 1: Hipotez testi yapılmadan önce varsayım kontrollerini yapınız.
# Bunlar Normallik Varsayımı ve Varyans Homojenliğidir. Kontrol ve test grubunun normallik varsayımına uyup uymadığını
# Purchase değişkeni üzerinden ayrı ayrı test ediniz.

# Normallik Varsayımı :
# H0: Normal dağılım varsayımı sağlanmaktadır. H1: Normal dağılım varsayımı sağlanmamaktadır.
# p < 0.05 H0 RED , p > 0.05 H0 REDDEDİLEMEZ
# Test sonucuna göre normallik varsayımı kontrol ve test grupları için sağlanıyor mu ? Elde edilen p-value değerlerini yorumlayınız.


test_stat, pvalue = shapiro(df.loc[df["durum"]=="test","Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# p-value > 0.05 -- can not be rejected

test_stat, pvalue = shapiro(df.loc[df["durum"]=="control","Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# p-value> 0.05 -- can not be rejected.

# Varyans Homojenliği :
# H0: Varyanslar homojendir.
# H1: Varyanslar homojen Değildir.
# p < 0.05 H0 RED , p > 0.05 H0 REDDEDİLEMEZ
# Kontrol ve test grubu için varyans homojenliğinin sağlanıp sağlanmadığını Purchase değişkeni üzerinden test ediniz.
# Test sonucuna göre normallik varsayımı sağlanıyor mu? Elde edilen p-value değerlerini yorumlayınız.

test_stat, pvalue = levene(df.loc[df["durum"]=="control","Purchase"],
                           df.loc[df["durum"]=="test","Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
#p-value > 0.05 --- can not be rejected.

# Adım 2: Normallik Varsayımı ve Varyans Homojenliği sonuçlarına göre uygun testi seçiniz.
# parametric test--> ttest_ind

# Adım 3: Test sonucunda elde edilen p_value değerini göz önünde bulundurarak kontrol ve test grubu satın alma
# ortalamaları arasında istatistiki olarak anlamlı bir fark olup olmadığını yorumlayınız.

test_stat, pvalue = ttest_ind(df.loc[df["durum"]=="control","Purchase"],
                           df.loc[df["durum"]=="test","Purchase"],
                              equal_var=True)
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# p-value>0.05 -- can not be rejected.




# Görev 4: Sonuçların Analizi
# Adım 1: Hangi testi kullandınız, sebeplerini belirtiniz.

# varsayımlar sağlanıyır-parametrik test

# Adım 2: Elde ettiğiniz test sonuçlarına göre müşteriye tavsiyede bulununuz.

# hipotez reddedilemedi. Ortalamalar arasında istatistiksel olarak anlamlı bir fark yoktur.
# Tavsiye: Daha çok veri ile daha kesin bir yargıya varabiliriz.














