import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules
from sklearn.preprocessing import Binarizer

# 1. Carregar os dados
df = pd.read_excel("dados/sfo_2018_data_file_final_Weightedv2.xlsx")

# 2. Selecionar e pré-processar as variáveis
df_subset = df[['NETPRO  ', 'Q23FLY', 'Q6LONGUSE']].copy()

# Discretizar as variáveis (exemplo de binarização)
df_subset['NETPRO_binned'] = (df_subset['NETPRO  '] > 8).astype(int)  # Satisfeito (1) ou Insatisfeito (0)
df_subset['Q23FLY_binned'] = (df_subset['Q23FLY'] > 5).astype(int)  # Alta Frequência de Voos (1) ou Baixa (0)
df_subset['Q6LONGUSE_binned'] = (df_subset['Q6LONGUSE'] > 3).astype(int)  # Usuário Longo (1) ou Novo (0)

# 3. Aplicar o FP-Growth
te = df_subset.astype(bool)  # Converter para booleano, necessário para o fpgrowth
frequent_itemsets = fpgrowth(te, min_support=0.1, use_colnames=True)

# 4. Gerar as regras de associação
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

# 5. Exibir algumas regras geradas
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
