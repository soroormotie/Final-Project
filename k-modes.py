import pandas as pd
import numpy as np
import scipy
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from kmodes.kmodes import KModes


df = pd.read_excel("sample.xlsx")

#----- dataframe dummies
df_with_dummies = pd.get_dummies(df)

print(df_with_dummies)

# random categorical data
data = np.random.choice(20, (100, 10))

km = KModes(n_clusters=4, init='Huang', n_init=5, verbose=1)

clusters = km.fit_predict(data)

# Print the cluster centroids
print(km.cluster_centroids_)

"""# define the k-modes model
km = KModes(n_clusters=10, init='Huang', n_init=11, verbose=1)
# fit the clusters to the skills dataframe
clusters = km.fit_predict(df_with_dummies)
# get an array of cluster modes
kmodes = km.cluster_centroids_
print(kmodes)

area = (20 * np.random.rand(50))**2
plt.scatter(df['seqhourportion'], df['genre'], s=area, alpha=0.5)
plt.scatter(kmodes[:, 0], kmodes[:, 1], c='red', s=50)
plt.show()
"""


#shape = kmodes.shape
"""
# For each cluster mode (a vector of "1" and "0")
# find and print the column headings where "1" appears.
# If no "1" appears, assign to "no-skills" cluster.
for i in range(shape[0]):
    if sum(kmodes[i,:]) == 0:
        print("\ncluster " + str(i) + ": ")
        print("no-skills cluster")
    else:
        print("\ncluster " + str(i) + ": ")
        cent = kmodes[i,:]
        for j in skills_df.columns[np.nonzero(cent)]:
            print(j)
"""


