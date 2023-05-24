import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

df = pd.read_excel('FrequentItemSet.xlsx', usecols=['ORDER_NUMBER', 'PRODUCT_NUMBER', 'product.1.PRODUCT_TYPE_CODE'])
transactions = df.groupby('ORDER_NUMBER')['product.1.PRODUCT_TYPE_CODE'].apply(list).tolist()
te = TransactionEncoder()
te_ary = te.fit_transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

# zoekt veelvoorkomende item sets met een minimum support van 0.01
frequent_itemsets = apriori(df, min_support=0.01, use_colnames=True)

# sorteer de top 10 op support
top_itemsets = frequent_itemsets.sort_values(by='support', ascending=False).head(10)

# genereerd een bar chart met de top 10
plt.figure(figsize=(10, 6))
sns.barplot(x='support', y='itemsets', data=top_itemsets, palette='viridis')
plt.title('Top 10 Frequente Itemsets')
plt.xlabel('Support')
plt.ylabel('Itemsets')
plt.show()

# maakt associaties met de itemsets met een confidence van 0.3
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.3)

#maken van scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.title('Association Rules')
plt.show()
