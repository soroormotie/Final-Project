import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder

dataset = pd.read_excel("reserve_1-cleaned.xlsx")

dataset = dataset [["genre"]]

#print(dataset)

#----- make all the types string
cols = list(dataset.columns)
for c in cols:
    dataset[c]=dataset[c].apply(lambda x : str(x))
#print(dataset.values)


#dataset = np.object0(dataset)

#print(dataset.dtypes)
#print(dataset.head())

#----- chnage dataframe to list
df_list = []
df_list = dataset.values.tolist()

#----- association rule preprocess

te = TransactionEncoder()
te_ary = te.fit(df_list).transform(df_list)
#print(te_ary)
df = pd.DataFrame(te_ary, columns=te.columns_)
#print(te.columns_)
df.to_excel("transform.xlsx")


from mlxtend.frequent_patterns import apriori, association_rules

frequent_itemsets = apriori(df, min_support=0.1, use_colnames=True)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
#frequent_itemsets.to_excel("frequent.xlsx")
#print(frequent_itemsets)
# min_threshold --> confidence
rules = association_rules(frequent_itemsets,metric= "confidence",min_threshold=0.1)
#print(type(rules))
rules.to_excel("rules.xlsx")



