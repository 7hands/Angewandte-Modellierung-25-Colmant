# Pakete laden
library(readr)
library(dplyr)
library(ggplot2)
library(cluster)
library(factoextra)

# Daten laden
wine <- read_csv("wine-quality-red.csv")

# Nur Features (ohne quality)
X <- wine %>% select(-quality)

# Skalieren
X_scaled <- scale(X)

# Elbow und Silhouette bestimmen
fviz_nbclust(X_scaled, kmeans, method = "wss") + ggtitle("Elbow Method")
fviz_nbclust(X_scaled, kmeans, method = "silhouette") + ggtitle("Silhouette Method")

# K-Means (beispielhaft k = 3)
km3 <- kmeans(X_scaled, centers = 3, nstart = 25)
wine$cluster_km <- factor(km3$cluster)

# Hierarchisches Clustering
d <- dist(X_scaled, method = "euclidean")
hc <- hclust(d, method = "ward.D2")
fviz_dend(hc, k = 3, rect = TRUE)

# Cluster-Zuweisung
wine$cluster_hc <- factor(cutree(hc, k = 3))

# Cluster-Zentren beschreiben
wine %>% group_by(cluster_km) %>% summarise_all(mean)
wine %>% group_by(cluster_hc) %>% summarise_all(mean)

# Visualisierung: PCA
fviz_cluster(list(data = X_scaled, cluster = km3$cluster))