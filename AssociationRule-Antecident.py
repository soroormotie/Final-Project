import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder

dataset = pd.read_excel("reserve_1-cleaned.xlsx")

dataset = dataset[["genre","count"]]

#dataset = np.random.choice(dataset.shape[0],int(0.3*dataset.shape[0],replace=False))


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

#------ based on one antecident
Event=df.columns[14:30]
EventConseq=[[x,rules[rules['antecedents']=={x}].loc[:,['antecedents','consequents','support','confidence','lift','leverage','conviction']]] for x in df.columns[14:30].tolist() if np.sum(rules['antecedents']=={x})>0]
ante_genre = EventConseq[19][1].iloc[:,:4]
print(ante_genre)

