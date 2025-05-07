import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# 1. Carregar os dados
df = pd.read_excel("dados/sfo_2018_data_file_final_Weightedv2.xlsx")

# 2. Selecionar colunas relevantes
colunas = ['NETPRO  ', 'Q20Age', 'Q21Gender', 'Q22Income',
           'Q23FLY', 'Q5TIMESFLOWN', 'Q6LONGUSE']
df_subset = df[colunas].copy()

# 3. Remover valores ausentes
df_clean = df_subset.dropna()

# 4. Criar variável binária de satisfação: 1 se NETPRO >= 9, senão 0
df_clean['satisfeito'] = df_clean['NETPRO  '].apply(lambda x: 1 if x >= 9 else 0)

# 5. Definir X e y
X = df_clean[['Q20Age', 'Q21Gender', 'Q22Income', 'Q23FLY', 'Q5TIMESFLOWN', 'Q6LONGUSE']]
y = df_clean['satisfeito']

# 6. Padronizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 7. Dividir dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)

# 8. Treinar modelo de Regressão Logística
model = LogisticRegression()
model.fit(X_train, y_train)

# 9. Avaliar o modelo
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("Relatório de Classificação:")
print(classification_report(y_test, y_pred))

print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))

print("ROC AUC Score:")
print(roc_auc_score(y_test, y_proba))

# 10. Coeficientes do modelo
coef_df = pd.DataFrame({
    'Variável': ['Q20Age', 'Q21Gender', 'Q22Income', 'Q23FLY', 'Q5TIMESFLOWN', 'Q6LONGUSE'],
    'Coeficiente': model.coef_[0]
})

print("\nCoeficientes do Modelo:")
print(coef_df)
