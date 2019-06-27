import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

dataset = pd.read_excel("sample.xlsx")


#print(dataset)

#----- make all the types string
cols = list(dataset.columns)
for c in cols:
    dataset[c]=dataset[c].apply(lambda x : str(x))


#----- chnage dataframe to list
df_list = []
df_list = dataset.values.tolist()

#----- association rule preprocess

te = TransactionEncoder()
te_ary = te.fit(df_list).transform(df_list)
df = pd.DataFrame(te_ary, columns=te.columns_)


supportLevels = [0.1, 0.05, 0.01, 0.005]
confidenceLevels = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

# Empty integers
rules_sup = np.arange(0,9)


for j in range(len(supportLevels)):
    for i in range(len(confidenceLevels)):
        frequent_itemsets = apriori(df, min_support=supportLevels[j], use_colnames=True,max_len=3)
        rules_sup[i] =len(association_rules(frequent_itemsets, metric="confidence",min_threshold=confidenceLevels[i]))
    #plt.subplot(2, 2, j + 1)
    plt.scatter(confidenceLevels , rules_sup)
    plt.plot(confidenceLevels , rules_sup)
    plt.show()
