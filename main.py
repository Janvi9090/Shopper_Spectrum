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






