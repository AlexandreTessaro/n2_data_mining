import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules
from sklearn.preprocessing import Binarizer
from tabulate import tabulate

# carregar os dados
df = pd.read_excel("dados/sfo_2018_data_file_final_Weightedv2.xlsx")

# selecionar e pre-processar as variaveis
df_subset = df[['NETPRO  ', 'Q23FLY', 'Q6LONGUSE']].copy()

# discretizar as variaveis 
df_subset['NETPRO_binned'] = (df_subset['NETPRO  '] > 8).astype(int)  # satisfeito (1) ou Insatisfeito (0)
df_subset['Q23FLY_binned'] = (df_subset['Q23FLY'] > 5).astype(int)  # alta Frequencia de Voos (1) ou Baixa (0)
df_subset['Q6LONGUSE_binned'] = (df_subset['Q6LONGUSE'] > 3).astype(int)  # usuario Longo (1) ou Novo (0)

# aplicar o FP-Growth
te = df_subset.astype(bool)  # converter para booleano, necessario para o fpgrowth
frequent_itemsets = fpgrowth(te, min_support=0.1, use_colnames=True)

# gerar as regras de associacao
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

# exibir algumas regras geradas com formatação aprimorada
rules_display = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]

# limitar a exibicao a um numero especifico de regras 
rules_display = rules_display.head(10)

# usando tabulate para uma saida mais legivel
print("\nRegras de Associação Geradas:")
print(tabulate(rules_display, headers='keys', tablefmt='fancy_grid', showindex=False))
