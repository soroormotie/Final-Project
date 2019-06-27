import pandas as pd
import numpy as np
import scipy
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df = pd.read_excel("reserve_1-cleaned.xlsx")
df_columns = df [["count"]]

df['Q'].loc[df['count']==1]='single'
df['Q'].loc[df['count']==2]='double'
df['Q'].loc[df['count']==3]='triple'
df['Q'].loc[df['count']==4]='quadruple'
df['Q'].loc[df['count']>=5]='more'

#----- dataframe dummies
df_with_dummies = pd.get_dummies(df_columns)

print(df_with_dummies.head(20))

"""
kmeans = KMeans(n_clusters=4).fit(df_with_dummies)
centroids = kmeans.cluster_centers_
print(centroids)


plt.scatter(df['sequencehour'], df['count'], c=kmeans.labels_.astype(float), s=50, alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
plt.show()
"""
