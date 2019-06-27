import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
df = pd.read_csv('seqhourportion-genre.csv')
print(df.dtypes)

#dummies
data = pd.get_dummies(df)
print(type(data))

#normalizing continuos features
"""
mms = MinMaxScaler()
mms.fit(data)
data_transformed = mms.transform(data)
"""

#Sum_of_squared

Sum_of_squared_distances = []
K = range(1,15)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(data)
    Sum_of_squared_distances.append(km.inertia_)

#elbow plot
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()

