import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configurações
input_path = "dados/sfo_2018_data_file_final_Weightedv2.xlsx"
output_path = "resultados/cluster_resultados.csv"
os.makedirs("resultados", exist_ok=True)

# 1. Carregar os dados
df = pd.read_excel(input_path)

# 2. Selecionar colunas relevantes
colunas = ['NETPRO  ', 'Q20Age', 'Q21Gender', 'Q22Income',
           'Q23FLY', 'Q5TIMESFLOWN', 'Q6LONGUSE']
df_subset = df[colunas].copy()

# 3. Remover linhas com valores ausentes
df_clean = df_subset.dropna()

# 4. Normalizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_clean)

# 5. Determinar o melhor número de clusters com método do cotovelo
inertia = []
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Exibir gráfico do cotovelo
plt.figure(figsize=(8, 5))
sns.lineplot(x=range(1, 10), y=inertia, marker='o')
plt.title('Método do Cotovelo')
plt.xlabel('Número de Clusters (K)')
plt.ylabel('Inércia')
plt.grid(True)
plt.savefig("resultados/grafico_cotovelo.png")
plt.close()

# 6. Aplicar KMeans com K=3
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df_clean['cluster'] = kmeans.fit_predict(X_scaled)

# 7. Tamanho relativo dos clusters
print("Distribuição dos clusters (%):")
print(df_clean['cluster'].value_counts(normalize=True) * 100)

# 8. Perfil médio por cluster
print("\nPerfil médio por cluster:")
print(df_clean.groupby('cluster').mean())

# 9. Salvar resultados
df_clean.to_csv(output_path, index=False)
