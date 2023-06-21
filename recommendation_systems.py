import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.metrics.pairwise import pairwise_distances
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import train_test_split

col_title = ['UserID', 'MovieID','Like', 'Timestamp'] # dataset icin basliklar
df = pd.read_csv('ml-100k/u.data', sep='\t',names=col_title)  # indirilen datasetin okunmasi

# aykiri deger kontrolu
# bar ile
value = df["Like"].value_counts()
value.plot(kind='bar',color='maroon')
plt.xlabel('Like')
plt.ylabel('Total')
# z skoru ile
stats.zscore(value)

# filmlerin begeni ortalamasinin alinmasi
mean_like = round(df.groupby(['MovieID'])['Like'].mean(),2)
total_like = df.groupby(['MovieID'])['Like'].count()
df_mean_col = ['Mean','Total']
df_info = pd.DataFrame([mean_like,total_like],df_mean_col).T
df_info.sort_values('Total',ascending=False)

# tavsiye yöntemi - işbirlikçi filtreleme
# kullanıcı-ürün matrisi
number_of_users = df.UserID.unique().shape[0]
number_of_movies = df.MovieID.unique().shape[0]
df_matrix = np.zeros((number_of_users,number_of_movies))
for r in df.itertuples(): # kullanıcı ve filmin kesiştiği beğeni sayılarının matrisi
    df_matrix[r[1]-1,r[2]-1]=r[3]
    
# kullanıcı tabanlı if
# benzerlik hesaplaması - kosinüs benzerliği
cos_similarity = pairwise_distances(df_matrix,metric="cosine")

# tahmin yapılması
def guess(rating,similarity):
    mean_rating = rating.mean(axis=1)
    dif = rating - mean_rating[:,np.newaxis]
    # .newaxis dizinin boyutunu arttırır
    # .dot skaler carpım için kullanılır
    # .abs mutlak değer alır
    guess =  mean_rating[:, np.newaxis] +similarity.dot(dif) / np.array([np.abs(similarity).sum(axis=1)]).T 
    return guess
  
# model tabanlı if
# veri kümesinin seyrekliği
rarity = round(1.0-len(df)/float(number_of_users*number_of_movies),4) # %93.7 seyrekliğe sahip
# rarity*100

# matris ayrıştırma - SVD
U, S, Vt = svds(df_matrix,k=30)

# RMSE değerlendirme ölçütü
def calculate_RMSE(guess,ref):
    guess = guess[ref.nonzero()].flatten()
    ref = ref[ref.nonzero()].flatten()
    return sqrt(mean_squared_error(guess,ref))
  
# test ve eğitim veri kümesinin ayrıştırılması
train_data, test_data = train_test_split(df,test_size = 0.30)

# eğitim verisi
df_train = np.zeros((number_of_users,number_of_movies))
for line in train_data.itertuples():
    df_train[line[1]-1,line[2]-1] = line[3]
    
# test verisi
df_test = np.zeros((number_of_users,number_of_movies))
for line in test_data.itertuples():
    df_test[line[1]-1,line[2]-1] = line[3]
    
# benzerlik 
cos_similar = pairwise_distances(df_train,metric="cosine")

# tahminleme
guess = guess(df_train,cos_similar)

# RMSE
print("Original RMSE:",calculate_RMSE(guess,df_test))

# SVD hesaplama
U,S,Vt = svds(df_train,k=30)
diag_matrix = np.diag(S)
guess_SVD = np.dot(np.dot(U,diag_matrix),Vt)
print("SVD RMSE:",calculate_RMSE(guess_SVD,df_test))
