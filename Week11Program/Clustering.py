import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_excel('OutdoorClustering.xlsx', sheet_name=0, nrows=50)
X = df[['COUNTRY_CODE', 'SEGMENT_CODE']]

# Drop lege samples zonder data
X = X.dropna()

#pak alleen de eerste 397 rijen
X = X.iloc[:397,:]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Labeled dots cluster method to estimate the number of clusters
scores = []
for n in range(2, 11):
    kmeans = KMeans(n_clusters=n, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(X_scaled)
    score = kmeans.score(X_scaled)
    scores.append(-score)
# voegt cluster indexes toe
kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=42)
kmeans.fit(X_scaled)
df['CLUSTER'] = kmeans.labels_

# reset index van X
X = X.reset_index(drop=True)

#x row
df = df.loc[X.index]
#visualiseren van clusters
plt.scatter(X['COUNTRY_CODE'], X['SEGMENT_CODE'], c=kmeans.labels_, cmap='rainbow', s=50, alpha=0.8)
plt.title('KMeans Clustering (k=4)')
plt.xlabel('COUNTRY_CODE')
plt.ylabel('SEGMENT_CODE')
plt.show()

