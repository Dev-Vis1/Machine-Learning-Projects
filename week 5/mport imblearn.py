#import imblearn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#from imblearn import  
from sklearn.cluster import KMeans



#dataset
from sklearn.datasets import make_blobs

X,y_true = make_blobs(n_samples=1000,centers=2, cluster_std=1,random_state=123)

from plotnine import ggplot, aes, geom_point

df = pd.DataFrame(X, columns=['x1','x2'])
df['y'] = y_true

#ggplot(df) + aes(x='x1', y='x2', color='y') + geom_point()

kmeans = KMeans(n_clusters=3,random_state=123)
kmeans.fit(X)

df_centers = pd.DataFrame(kmeans.cluster_centers_, columns=['x1','x2'])
df_centers

y_kmeans = kmeans.predict(X)
df['y_kmeans'] = y_kmeans

#(ggplot(df) + aes(x='x1', y='x2', color='y') + geom_point() + geom_point(data=df_centers, color='black',size=3))