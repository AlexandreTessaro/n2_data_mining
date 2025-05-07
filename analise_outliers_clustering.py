import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tabulate import tabulate

# configurações
input_path = "dados/sfo_2018_data_file_final_Weightedv2.xlsx"
output_path = "resultados/cluster_resultados.csv"
os.makedirs("resultados", exist_ok=True)

# carregar os dados
df = pd.read_excel(input_path)

# selecionar colunas relevantes
colunas = ['NETPRO  ', 'Q20Age', 'Q21Gender', 'Q22Income',
           'Q23FLY', 'Q5TIMESFLOWN', 'Q6LONGUSE']
df_subset = df[colunas].copy()

# remover linhas com valores ausentes
df_clean = df_subset.dropna()

# normalizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_clean)

# determinar o melhor numero de clusters com metodo do cotovelo
inertia = []
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# exibir grafico do cotovelo
plt.figure(figsize=(8, 5))
sns.lineplot(x=range(1, 10), y=inertia, marker='o')
plt.title('Método do Cotovelo')
plt.xlabel('Número de Clusters (K)')
plt.ylabel('Inércia')
plt.grid(True)
plt.savefig("resultados/grafico_cotovelo.png")
plt.close()

# aplicar KMeans com K=3
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df_clean['cluster'] = kmeans.fit_predict(X_scaled)

# tamanho relativo dos clusters com formatação melhorada e bordas
cluster_distribution = df_clean['cluster'].value_counts(normalize=True) * 100
cluster_distribution_formatted = cluster_distribution.reset_index()
cluster_distribution_formatted.columns = ['Cluster', 'Proporção (%)']
print("Distribuição dos clusters (%):")
print(tabulate(cluster_distribution_formatted, headers='keys', tablefmt='fancy_grid', showindex=False))

# perfil medio por cluster com formatacao de casas decimais e bordas
print("\nPerfil médio por cluster:")
profile_mean = df_clean.groupby('cluster').mean()
print(tabulate(profile_mean, headers='keys', tablefmt='fancy_grid', showindex=True))

# salvar resultados
df_clean.to_csv(output_path, index=False)
