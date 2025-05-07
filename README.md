# Projeto de Mineração de Dados - N2

Este projeto aborda a análise de dados usando técnicas de Mineração de Dados para resolver três problemas distintos com base em um dataset de passageiros do aeroporto de São Francisco (SFO). As técnicas utilizadas incluem **Clustering**, **Regras de Associação** e **Regressão Logística**.

## Estrutura do Projeto

Este projeto está dividido em três partes principais:

1.  **Parte 1 - Análise de Anomalias / Outliers com Clustering**
    
2.  **Parte 2 - Regras de Associação em Problema Não-Comercial**
    
3.  **Parte 3 - Regressão Logística como Alternativa à Linear**
    

----------

## Parte 1 - Análise de Anomalias / Outliers com Clustering

### Objetivo

Verificar a existência de um grupo incomum de passageiros (outliers) no dataset `sfo_2018_data_file_final_Weightedv2.xlsx`.

### Passos Realizados

1.  Carregamento e limpeza do dataset (remoção de valores ausentes).
    
2.  Seleção das variáveis: `NETPRO`, `Q20Age`, `Q21Gender`, `Q22Income`, `Q23FLY`, `Q5TIMESFLOWN`, `Q6LONGUSE`.
    
3.  Normalização dos dados usando `StandardScaler`.
    
4.  Determinação do número ideal de clusters com o método do cotovelo (K=3).
    
5.  Aplicação do algoritmo K-Means.
    
6.  Identificação do cluster incomum e análise do perfil médio.
    

### Resultados

-   **Cluster Incomum Identificado**: Cluster 1
    
-   **Tamanho**: 6,19% dos passageiros
    
-   **Perfil Médio do Cluster 1**:
    
    -   `NETPRO`: 9.75
        
    -   Idade (`Q20Age`): 0.73
        
    -   Gênero (`Q21Gender`): 0.23
        
    -   Renda (`Q22Income`): 0.11
        
    -   Frequência de voo (`Q23FLY`): 0.18
        
    -   Experiência de voo (`Q5TIMESFLOWN`): 2.35
        
    -   Tempo de uso do aeroporto (`Q6LONGUSE`): 2.54
        

----------

## Parte 2 - Regras de Associação em Problema Não-Comercial

### Objetivo

Identificar padrões de comportamento entre os usuários do aeroporto com base em hábitos e satisfação.

### Passos Realizados

1.  Seleção das variáveis: `NETPRO`, `Q6LONGUSE`, `Q23FLY`.
    
2.  Pré-processamento com discretização (binning) das variáveis numéricas.
    
3.  Conversão para formato transacional (one-hot encoding).
    
4.  Aplicação do algoritmo **FP-Growth** com suporte mínimo de 20%.
    

### Exemplos de Regras Geradas

1.  Se `Q6LONGUSE`, então `NETPRO` — Suporte: 96.69%, Confiança: 99.27%, Lift: 1.00
    
2.  Se `NETPRO`, então `Q6LONGUSE` — Suporte: 96.69%, Confiança: 97.52%, Lift: 1.00
    
3.  Se `NETPRO`, então `Q23FLY` — Suporte: 92.27%, Confiança: 93.07%, Lift: 1.00
    
4.  Se `Q23FLY`, então `NETPRO` — Suporte: 92.27%, Confiança: 99.20%, Lift: 1.00
    

----------

## Parte 3 - Regressão Logística como Alternativa à Linear

### Objetivo

Prever se um passageiro está satisfeito (NETPRO >= 9) com base em variáveis demográficas e comportamentais.

### Passos Realizados

1.  Criação da variável alvo binária (1 se `NETPRO >= 9`, caso contrário 0).
    
2.  Separação dos dados em treino e teste (75%/25%).
    
3.  Normalização dos dados.
    
4.  Treinamento do modelo com **Logistic Regression**.
    
5.  Avaliação do modelo com métricas como **acurácia**, **f1-score**, **matriz de confusão** e **ROC AUC**.
    

### Resultados

-   **Acurácia**: 55%
    
-   **ROC AUC**: 0.55
    
-   **Matriz de Confusão**:
    
    lua
    
    CopiarEditar
    
    `[[179 173]
     [144 207]]` 
    
-   **Coeficientes do Modelo**:
    
    -   `Q20Age`: -0.07
        
    -   `Q21Gender`: 0.02
        
    -   `Q22Income`: -0.16
        
    -   `Q23FLY`: -0.16
        
    -   `Q5TIMESFLOWN`: 0.02
        
    -   `Q6LONGUSE`: 0.23
        

----------

## Ferramentas Utilizadas

-   **Python** (versão 3.x)
    
-   **Bibliotecas**:
    
    -   `pandas`: Manipulação de dados.
        
    -   `sklearn`: Algoritmos de clustering, regressão logística e métricas de avaliação.
        
    -   `matplotlib`, `seaborn`: Visualização de gráficos.
        
    -   `mlxtend`: Algoritmo FP-Growth para regras de associação.
        

----------

## Como Executar o Projeto

1.  **Instalar dependências**:  
    Crie e ative um ambiente virtual (opcional) e instale as dependências com o seguinte comando:
    
    `pip install -r requirements.txt` 
    
2.  **Executar cada parte do projeto**:
    
    -   **Parte 1**: Execute o script `analise_outliers_clustering.py`.
        
    -   **Parte 2**: Execute o script `analise_regras_associacao.py`.
        
    -   **Parte 3**: Execute o script `modelo_regressao_logistica.py`.
        
3.  **Resultados**:
    
    -   Os resultados de cada parte serão salvos em arquivos CSV ou exibidos no terminal.
        
    -   Gráficos gerados serão salvos como imagens PNG.
        

----------

## Observações

-   O dataset utilizado foi o `sfo_2018_data_file_final_Weightedv2.xlsx`.
    
-   A análise foi realizada com base em variáveis demográficas, comportamentais e de satisfação dos passageiros.
    
-   Os modelos de aprendizado de máquina (clustering e regressão) podem ser ajustados para obter melhores resultados.
    

----------

## Contato

Se você tiver alguma dúvida ou sugestão, sinta-se à vontade para entrar em contato!

-   **Autor**: Alexandre Tessaro