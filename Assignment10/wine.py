import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_wine
import pandas as pd

# sklearn‑Wein‑Datensatz laden
data = load_wine(as_frame=True)

# DataFrame mit Features …
X = data.data
# … und Zielvariable
y = data.target.rename("quality")  # wir nennen das Target "quality" zum Vergleich

# Optional: alles in einem DataFrame
wine = pd.concat([X, y], axis=1)

# Skalieren
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow und Silhouette
inertia = []
sil = []
K = range(2, 10)
for k in K:
    km = KMeans(n_clusters=k, random_state=42).fit(X_scaled)
    inertia.append(km.inertia_)
    sil.append(silhouette_score(X_scaled, km.labels_))

plt.figure()
plt.plot(K, inertia, marker='o')
plt.title("Elbow Method")
plt.xlabel("k"); plt.ylabel("Inertia")
plt.show()

plt.figure()
plt.plot(K, sil, marker='o')
plt.title("Silhouette Score")
plt.xlabel("k"); plt.ylabel("Silhouette")
plt.show()

# KMeans mit k=3
km3 = KMeans(n_clusters=3, random_state=42).fit(X_scaled)
wine['cluster_km'] = km3.labels_

# Hierarchisch
linked = linkage(X_scaled, method='ward')
plt.figure(figsize=(10, 5))
dendrogram(linked, truncate_mode='level', p=5)
plt.title("Hierarchical Clustering Dendrogram")
plt.show()

hc3 = AgglomerativeClustering(n_clusters=3, linkage='ward').fit(X_scaled)
wine['cluster_hc'] = hc3.labels_

# Cluster-Mittelwerte
print(wine.groupby('cluster_km').mean())
print(wine.groupby('cluster_hc').mean())

# PCA-Visualisierung
pca = PCA(n_components=2)
proj = pca.fit_transform(X_scaled)
plt.figure()
for label in np.unique(km3.labels_):
    mask = km3.labels_ == label
    plt.scatter(proj[mask,0], proj[mask,1], label=f"KM {label}")
plt.legend()
plt.title("PCA Projektion KMeans")
plt.show()