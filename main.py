import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('online_retail.csv')
print(data.shape)
# print(data.columns)

print("Missing values: ")
print(data.isnull().sum())
data = data.dropna(subset=['Description'])
data = data.dropna(subset=['CustomerID'])

print(data.shape)
print("Missing values: ")
print(data.isnull().sum())

# Remove cancelled orders
data = data[~data['InvoiceNo'].astype(str).str.startswith('C')]
#Remove invalid quantity and price
data = data[(data['Quantity']>0) & (data['UnitPrice']>0)]
#Fix data types
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
data['CustomerID'] = data['CustomerID'].astype(int)

#EDA
#Transactionby Country
# plt.figure(figsize=(10,5))
# data['Country'].value_counts().head(10).plot(kind='bar')
# plt.title("Top 10 countries by transactions")
# plt.show()

#Top Selling Products
# top_products = data.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)
# plt.figure(figsize=(10,5))
# top_products.plot(kind='bar')
# plt.title("Top 10 selling Products")
# plt.show()

# data['Month'] = data['InvoiceDate'].dt.to_period('M')
# print(data[['InvoiceDate', 'Month']].head())

#Purchase Trend Over Time
# data['Month'] = data['InvoiceDate'].dt.to_period('M')
# monthly_sales = data.groupby('Month').size()
# plt.figure(figsize=(12,5))
# monthly_sales.plot()
# plt.title("Monthly Purchase Trend")
# plt.show()

#Monetry Distribution
# data['TotalPrice'] = data['Quantity'] * data['UnitPrice']
# plt.figure(figsize=(8,5))
# sns.histplot(data['TotalPrice'], bins=50)
# plt.title("Transaction value distribution")
# plt.show()

#RFM Feature Engineering
latest_date = data['InvoiceDate'].max()
data['TotalPrice'] = data['Quantity'] * data['UnitPrice']

rfm = data.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (latest_date - x.max()).days,
    'InvoiceNo': 'count',
    'TotalPrice': 'sum'
}).reset_index()

rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
rfm.head()

#RFM Scaling
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

#Find Optimal Cluster(Elbow+ Silhouette)
inertia = []
for k in range(2, 10):
    km = KMeans(n_clusters= k, random_state=42)
    km.fit(rfm_scaled)
    inertia.append(km.inertia_)
plt.figure(figsize=(8,5))
plt.plot(range(2, 10), inertia, marker='o')
plt.title("Elbow Method")
plt.show()

#Silhouette Score
for k in range(2, 6):
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(rfm_scaled)
    print(f"K={k}, Silhouette Score={silhouette_score(rfm_scaled, labels):.3f}")

#Train using Kmeans
kmeans = KMeans(n_clusters=4, random_state=42)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

cluster_summary = rfm.groupby('Cluster').mean()
print(cluster_summary)

#Labels
cluster_labels= {
    0: 'Regular',
    1: 'Occasional',
    2: 'High_Value',
    3: 'At-Risk'
}

rfm['segment'] = rfm['Cluster'].map(cluster_labels)

#Visualize Clusters
plt.figure(figsize=(8,5))
sns.scatterplot(x='Recency', y='Monetary', hue= 'segment', data=rfm)
plt.title("Customer Segmentation")
plt.show()

#Product Recommendation System
#Customer-Product Matrix
product_matrix = data.pivot_table{
    index= 'CustomerID',
    columns= 'Description',
    values='Quantity',
    aggfunc='sum',
    fill_value= 0
    }


