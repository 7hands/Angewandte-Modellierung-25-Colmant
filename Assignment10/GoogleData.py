import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# --- 0) Arbeitsverzeichnis prüfen ---
def check_working_directory():
    print("Working directory:", os.getcwd())
    print("Files in directory:")
    for f in sorted(os.listdir(os.getcwd())):
        print("  -", f)

check_working_directory()

# --- 1) Daten einlesen und Spalten anpassen ---
# Klima-Monatsdaten
klima = pd.read_csv('KlimaDaten.csv', parse_dates=['Monat'])
klima = klima.rename(columns={'Monat': 'Date', 'Klima': 'Value_Klima'})
klima['Date'] = klima['Date'].dt.to_period('M').dt.to_timestamp()

# SchweineBauch-Monatsdaten
schwein = pd.read_csv('SchweineBauchData.csv', parse_dates=['Monat'])
schwein = schwein.rename(columns={'Monat': 'Date', 'Schweinebauch': 'Value_Schwein'})
schwein['Date'] = schwein['Date'].dt.to_period('M').dt.to_timestamp()

# Faschos-Zeitdaten
fasch_time = pd.read_csv('FaschosÜberZeit.csv', parse_dates=['Monat'])
fasch_time = fasch_time.rename(columns={'Monat': 'Date', 'Faschos': 'Value_Faschos'})
fasch_time['Date'] = fasch_time['Date'].dt.to_period('M').dt.to_timestamp()

# LOL- und Valorant-Wochendaten
def load_weekly(fname):
    df = pd.read_csv(fname)
    df = df.rename(columns={df.columns[0]: 'Date', df.columns[1]: 'Value'})
    df['Date'] = pd.to_datetime(df['Date'])
    return df

lol = load_weekly('LOLGoogleData.csv').rename(columns={'Value': 'Value_LOL'})
valorant = load_weekly('ValorantGoogleData.csv').rename(columns={'Value': 'Value_Valorant'})

# Geo-Daten: Klima & Schwein & Faschos
klima_geo = pd.read_csv('KlimaGeo.csv').rename(columns={'Klima': 'Value_Klima'})
schwein_geo = pd.read_csv('SchweinGeo.csv').rename(columns={'Schweinebauch': 'Value_Schwein'})
fasch_land = pd.read_csv('FaschosÜberLand.csv').rename(columns={'Faschos': 'Value_Faschos'})

geo = pd.merge(pd.merge(klima_geo, schwein_geo, on='Region'), fasch_land, on='Region')

# --- 2) Zeitreihen-Vergleich mit Partei-Daten ---
time_df = klima[['Date', 'Value_Klima']]
time_df = time_df.merge(schwein[['Date', 'Value_Schwein']], on='Date', how='outer')
time_df = time_df.merge(fasch_time[['Date', 'Value_Faschos']], on='Date', how='outer')

plt.figure(figsize=(10, 4))
for col, label in [('Value_Klima','Klima'),('Value_Schwein','Schweinebauch'),('Value_Faschos','Faschos')]:
    plt.plot(time_df['Date'], time_df[col], label=label)
plt.title('Trends: Klima, Schweinebauch & Partei Faschos')
plt.xlabel('Datum (Monat)')
plt.ylabel('Normiertes Interesse / Unterstützung')
plt.legend()
plt.tight_layout()
plt.show()

# --- 3) Zeitreihen-Vergleich: LOL vs. Valorant ---
plt.figure(figsize=(10, 4))
plt.plot(lol['Date'], lol['Value_LOL'], label='LOL')
plt.plot(valorant['Date'], valorant['Value_Valorant'], label='Valorant')
plt.title('Google Trends: LOL vs. Valorant')
plt.xlabel('Datum (Woche)')
plt.ylabel('Normiertes Suchinteresse')
plt.legend()
plt.tight_layout()
plt.show()

# --- 4) Geographische Analyse: KMeans und PCA ---
# Feature-Matrix
features = ['Value_Klima','Value_Schwein','Value_Faschos']
X = geo[features]

# 4a) KMeans-Cluster
sil_scores = {}
for k in range(2,6):
    km = KMeans(n_clusters=k, random_state=42).fit(X)
    sil_scores[k] = silhouette_score(X, km.labels_)
best_k = max(sil_scores, key=sil_scores.get)
print('Optimale Clusterzahl (Silhouette):', best_k)
kmeans = KMeans(n_clusters=best_k, random_state=42).fit(X)
geo['Cluster'] = kmeans.labels_

# 4b) PCA auf drei Dimensionen -> 2 Komponenten
pca = PCA(n_components=2, random_state=42)
pca_coords = pca.fit_transform(X)
geo['PC1'], geo['PC2'] = pca_coords[:,0], pca_coords[:,1]
print('Erklärte Varianz PC1, PC2:', pca.explained_variance_ratio_)

# Scatter der PCA-Komponenten, gefärbt nach Cluster
plt.figure(figsize=(8,6))
scatter = plt.scatter(
    geo['PC1'], geo['PC2'],
    c=geo['Cluster'], cmap='tab10', s=100, edgecolor='k'
)
for _, row in geo.iterrows():
    plt.text(row['PC1']+0.02, row['PC2']+0.02, row['Region'], fontsize=8)
plt.title('PCA (2D) der Geo-Daten mit KMeans-Clustern')
plt.xlabel('PC1')
plt.ylabel('PC2')
cbar = plt.colorbar(scatter, ticks=range(best_k))
cbar.set_label('Cluster')
plt.tight_layout()
plt.show()

# 4c) Original 2D-Scatter (Klima vs Schwein) mit Farbe Faschos
plt.figure(figsize=(8, 6))
scatter2 = plt.scatter(
    geo['Value_Klima'], geo['Value_Schwein'],
    c=geo['Value_Faschos'], cmap='viridis', s=100, edgecolor='k'
)
for _, row in geo.iterrows():
    plt.text(row['Value_Klima']+0.5, row['Value_Schwein']+0.5, row['Region'], fontsize=8)
plt.title('Geo-Vergleich: Klima vs. Schwein (Farbe = Faschos)')
plt.xlabel('Interesse Klima')
plt.ylabel('Interesse Schweinebauch')
cbar2 = plt.colorbar(scatter2)
cbar2.set_label('Unterstützung Faschos')
plt.tight_layout()
plt.show()

# Ausgabe der Geo-Tabelle
print(geo)