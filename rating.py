###################################################
# Rating Products
###################################################

# - Average
# - Time-Based Weighted Average
# - User-Based Weighted Average
# - Weighted Rating

#Bu bölümde amaç : Bir ürüne verilen puanlar üzerinden çeşitli değerlendirmeler yaparak
# en doğru puanın nasıl hesaplanabileceğine dair bir uygulama yapmak

############################################
# Uygulama: Kullanıcı ve Zaman Ağırlıklı Kurs Puanı Hesaplama
############################################

import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

# (50+ Saat) Python A-Z™: Veri Bilimi ve Machine Learning
# Puan: 4.8 (4.764925)
# Toplam Puan: 4611
# Puan Yüzdeleri: 75, 20, 4, 1, <1
# Yaklaşık Sayısal Karşılıkları: 3458, 922, 184, 46, 6



#%75 5 puan verilmiş, %20 4 puan, %4 3 puan, %1 2 puan, %1'den daha az 1 puan verilmiş
#bu şekilde bir puanlama yapılmış, verinin kendisi elimizde olduğundan dolayı bir inceleme yapacağız

df = pd.read_csv(r"C:\Users\MerveATASOY\Desktop\data_scientist_miuul\egitim_teorik_icerikler\Bolum_5_Measurement_Problems\datasets\course_reviews.csv")
df.head()
df.shape

# rating dagılımı
df["Rating"].value_counts()
#yukarıda verilen bilgiyle bu örtüşmeyebilir, birbirine yakın olabilir ama örtüşmesini beklemeyiz çünkü bu tabloda puaanlar 5''e bölünmüş durumda ama üst kısımda küsüratlarda vardır

df["Questions Asked"].value_counts()  #sorulan soruların dağılımı

#soruların soru kırılımında verilen puan nedir
df.groupby("Questions Asked").agg({"Questions Asked": "count",
                                   "Rating": "mean"})


df.head()

####################
# Average
####################

# Ortalama Puan
df["Rating"].mean()

#ilgili ürün ya da eğitim ile ilgili müşteriler açısından son zamanlardaki trendi kaçırıyor olabiliriz, yani memnuniyet trendini kaçırabiliyor oluruz
#son zamanlarda ürün ile ilgili ortaya çıkma ihtimali olan bazı problemler olabilir dolayısıyla bir şekilde ürünün sunumu ya da ele alınımıyla ilgili ortaya çıkabilecek pozitif ya da negatif
#trendler etkisini yitirebiliyor olacaktır.

#Örneğin bir ürünün ilk üç ayından çok ciddi yüksek puanlamalar alınması durumu ağırlığını diğerleriylee aynı etkide hissetiriyor olduğundan dolayı son zamanlardaki daha az ya da daha çok beğenme trendi kaçıyor olacaktır.
#dolayısıyla sadece bir puan ortamalası almak yerine başka şeylerde yapmak gerekir

#Sizce ne yaparsak güncel trendi ortalamaya daha iyi bir şekilde yansıtabiliriz ?

####################
# Time-Based Weighted Average
####################
# Puan Zamanlarına Göre Ağırlıklı Ortalama


df.head()
df.info()

df["Timestamp"] = pd.to_datetime(df["Timestamp"])

#yapılan yorumları gün cinsinden ifade etmeliyiz, current date lazım = veri setindeki maximum tarih
current_date = pd.to_datetime('2021-02-10 0:0:0')

df["days"] = (current_date - df["Timestamp"]).dt.days

df.loc[df["days"] <= 30, "Rating"].mean()   #30 gün ve aşağısındaki toplam puanların ortalaması

df.loc[(df["days"] > 30) & (df["days"] <= 90), "Rating"].mean() #30 ve 90 gün arasındaki toplam puanların ortalaması

df.loc[(df["days"] > 90) & (df["days"] <= 180), "Rating"].mean() #90 ile 180 gün arasındaki toplam puanların ortalaması

df.loc[(df["days"] > 180), "Rating"].mean()   #180 günden büyük toplam puanların ortalaması

#sonuçlara bakıldığında son zamanlarda bu kursun memnuniyetiyle alakalı bir artış olduğu görülecektir

#elde edilen sonuçlar için farklı ağırlıklar verilerek zamanın etkisi ağırlık hesabına yansıtılabilir


#aşağı satırdan kod yazmaya devam etmek için \ kullanıldı
#2. kısımın ağırlığına 26 verildi ilk kısımdan zaman açısından önemi daha düşük olduğu için

#burada dikkat edilmesi gereken şey tüm verilen ağırlıkların toplamı 100 olmalı
df.loc[df["days"] <= 30, "Rating"].mean() * 28/100 + \
    df.loc[(df["days"] > 30) & (df["days"] <= 90), "Rating"].mean() * 26/100 + \
    df.loc[(df["days"] > 90) & (df["days"] <= 180), "Rating"].mean() * 24/100 + \
    df.loc[(df["days"] > 180), "Rating"].mean() * 22/100

#4.76 olarak hesaplandı, kursun puani ile hemen hemen aynı, demekki oradada böyle birşey yapılmış

def time_based_weighted_average(dataframe, w1=28, w2=26, w3=24, w4=22):
    return dataframe.loc[df["days"] <= 30, "Rating"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["days"] > 30) & (dataframe["days"] <= 90), "Rating"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["days"] > 90) & (dataframe["days"] <= 180), "Rating"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["days"] > 180), "Rating"].mean() * w4 / 100

time_based_weighted_average(df)

time_based_weighted_average(df, 30, 26, 22, 22)

#bir ürünün sıralanamasında skor hesabı yaparken 4. basamakta sıralama değişebilir,  değişen sıralamada ürün başına 10bin belkide 100 bin fazla tıklama alabilir ve bu tıklamalar neticesinde ciddi miktarlarda gelir artışları olabilir
#bilimsel bir çalışma alanında olduğumuzdan dolayı virgülden sonrası zaten önemlidir ama buradada ayrı bir önemi vardır

####################
# User-Based Weighted Average
####################
#acaba bütün kullanıcıların verdiği puanlar aynı ağırlığa mı sahip olmalı?
#yani örneğin kursun tamamını izleyen ile %1'ni izleyen kişi aynı ağırlığa mı sahip olmalı

#acaba kursun izlenme oranlarına göre daha farklı bir ağırlık mı olmalı ?
# bu durum user quality olarak birçok sektörde karşımıza gelebilir
#örnek verecek olursak imdb'yi düşünelim burada top 250 listeleri var; buraya gelip üye olup ertesi gün puanlama yapan birisinin puanı daha önce belkide yüzlerce, binlerce filme puan vermiş, yorum yapmış kişinin puanıyla anı değildir
#birçok uluslararası online satış platformunda yapılan yorumların öne çıkarılmasında da aynı durum söz konusudur
#örneğin bir senaryoda kullanıcı bir ürün almış 5 puanlık pozitif bir sürü yorum yapmış  burada bazı satekarlıkların ve aldatma çabalarının önüne geçmek, sosyal ispat sunumunu en doğru şekilde yapabilmek için
#kullanılan genel yöntemlerden bir tanesidir, bu iş probleminde de izlenme oranı user quality,user score, user rank gibi ya da user based gibi isimlendirilecek bu yaklaşımlarla dikkate alınmalıdır


df.head()

#ilerlemeye göre grupby'a alınıp ortalamaya bakılabilir
df.groupby("Progress").agg({"Rating": "mean"})
#bu tablodan ilerleme durumuna gör artış var gibi duruyor


df.loc[df["Progress"] <= 10, "Rating"].mean() * 22 / 100 + \
    df.loc[(df["Progress"] > 10) & (df["Progress"] <= 45), "Rating"].mean() * 24 / 100 + \
    df.loc[(df["Progress"] > 45) & (df["Progress"] <= 75), "Rating"].mean() * 26 / 100 + \
    df.loc[(df["Progress"] > 75), "Rating"].mean() * 28 / 100

#kursu tamamen izleyen kişi kursu tanımıştır bu kişinin vereceği puan ağırlığı ile diğerlerinin vereceği puan ağırlığı aynı olmamalıdır
#bu kısım yorum değil sektörde yaygınca uygulanan bir durumdur

def user_based_weighted_average(dataframe, w1=22, w2=24, w3=26, w4=28):
    return dataframe.loc[dataframe["Progress"] <= 10, "Rating"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["Progress"] > 10) & (dataframe["Progress"] <= 45), "Rating"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["Progress"] > 45) & (dataframe["Progress"] <= 75), "Rating"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["Progress"] > 75), "Rating"].mean() * w4 / 100


user_based_weighted_average(df, 20, 24, 26, 30)

#time-based ve user-based'ın ikisinin birden ağırlıklı olarak ortalaması alınırsa daha anlamlı olur

####################
# Weighted Rating
####################

def course_weighted_rating(dataframe, time_w=50, user_w=50):
    return time_based_weighted_average(dataframe) * time_w/100 + user_based_weighted_average(dataframe)*user_w/100

course_weighted_rating(df)

course_weighted_rating(df, time_w=40, user_w=60)

#time_basede ön tanımlı %50, user-based'a ön tanımlı %50 ağırlıklandırma yapılarak skor hesaplandı








