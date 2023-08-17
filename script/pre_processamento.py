from utils import *

# Carregando dados
dados = pd.read_csv("C:/Projetos Pessoais/DataScience/analise_Previsao_Churn_teleComunicacoes/dados/dados_eng")
dados_copy = dados.copy()
print(dados.head())

''' Pré Processamento de variáveis categóricas com One Hot Encoding e Label Encoding '''

"""
Label Encoding para variável target - Churn
"""

print(dados.nunique())
print(dados.Churn.value_counts())

# Criando objeto Encoder e realizar treino
le = LabelEncoder()
le.fit(dados.Churn)
print(le.classes_)

# Aplica objeto encoder
dados.Churn = le.transform(dados.Churn)
print(dados.Churn.value_counts())
print(dados.sample(4))

"""
Aplicando One Hot Encoding nas categóricas nominais de entrada
"""
#Alterando variavel 'idoso' para str() com 'Yes' e 'No' para aplicar o encoding
dados['idoso'] = dados['idoso'].map(map_idoso)

print(dados.dtypes)

# Aplicando One-Hot Encoding
for cat in cats:
    onehots = pd.get_dummies(dados[cat], prefix=cat)
    dados = dados.join(onehots)

# Dropar as colunas que não serão mais uteis
dados = dados.drop(columns = cats)

# Os dados ficaram no formato True/False - transformar em 0 e 1
for x in variaveis_processadas:
    dados[x] = np.where(dados[x] == True, 1,0)


''' Pré Processamento de variáveis numéricas - Padronização'''

for n in nums_valores:
    dados[n] = StandardScaler().fit_transform(dados['valor_mensal'].values.reshape(len(dados), 1))

print(dados.sample(10))




