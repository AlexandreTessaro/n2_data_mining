import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# carregar os dados
df = pd.read_excel("dados/sfo_2018_data_file_final_Weightedv2.xlsx")

# selecionar colunas relevantes
colunas = ['NETPRO  ', 'Q20Age', 'Q21Gender', 'Q22Income',
           'Q23FLY', 'Q5TIMESFLOWN', 'Q6LONGUSE']
df_subset = df[colunas].copy()

# remover valores ausentes
df_clean = df_subset.dropna()

# criar variavel binaria de satisfacao: 1 se NETPRO >= 9, senao 0
df_clean['satisfeito'] = df_clean['NETPRO  '].apply(lambda x: 1 if x >= 9 else 0)

# definir X e y
X = df_clean[['Q20Age', 'Q21Gender', 'Q22Income', 'Q23FLY', 'Q5TIMESFLOWN', 'Q6LONGUSE']]
y = df_clean['satisfeito']

# padronizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# dividir dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)

# treinar modelo de Regressao Logistica
model = LogisticRegression()
model.fit(X_train, y_train)

# avaliar o modelo
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# relatorio de Classificacao
print("\n----------------------- Relatório de Classificação -----------------------")
report = classification_report(y_test, y_pred, target_names=['Classe 0', 'Classe 1'], digits=2)
print(report)

# matriz de Confusao
print("\n----------------------- Matriz de Confusão ------------------------------")
cm = confusion_matrix(y_test, y_pred)
print(f"[[{cm[0][0]} {cm[0][1]}]  # Verdadeiros Negativos (0) e Falsos Positivos (1)]")
print(f" [{cm[1][0]} {cm[1][1]}]]  # Falsos Negativos (0) e Verdadeiros Positivos (1)")

# ROC AUC Score
roc_auc = roc_auc_score(y_test, y_proba)
print(f"\n----------------------- ROC AUC Score ------------------------------")
print(f"ROC AUC Score: {roc_auc:.4f}")

# coeficientes do Modelo
print("\n----------------------- Coeficientes do Modelo ------------------------")
coef_df = pd.DataFrame({
    'Variável': ['Q20Age', 'Q21Gender', 'Q22Income', 'Q23FLY', 'Q5TIMESFLOWN', 'Q6LONGUSE'],
    'Coeficiente': model.coef_[0]
})
print(coef_df.to_string(index=False))
