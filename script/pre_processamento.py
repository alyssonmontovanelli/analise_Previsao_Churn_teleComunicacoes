from utils import *

# Carregando dados
dados = pd.read_csv("C:/Projetos Pessoais/DataScience/analise_Previsao_Churn_teleComunicacoes/dados/dados_eng")
dados_copy = dados.copy()


"""
Label Encoding para variável target - Churn
"""

# Criando objeto Encoder e realizar treino
le = LabelEncoder()
le.fit(dados.Churn)

# Aplica objeto encoder
dados.Churn = le.transform(dados.Churn)


"""
Aplicando One Hot Encoding nas categóricas nominais de entrada
"""
# Alterando variavel 'idoso' para str() com 'Yes' e 'No' para aplicar o encoding
# Alterando variavel 'fidelidade' para categoria
dados['idoso'] = dados['idoso'].map(map_idoso)
# dados['fidelidade'] = dados['fidelidade'].map(map_fidelidade)

# Aplicando One-Hot Encoding
for cat in cats:
    onehots = pd.get_dummies(dados[cat], prefix=cat)
    dados = dados.join(onehots)

# Dropar as colunas que não serão mais uteis
dados = dados.drop(columns = cats)

# Os dados ficaram no formato True/False - transformar em 0 e 1
for x in variaveis_processadas:
    dados[x] = np.where(dados[x] == True, 1,0)

"""
 Pré Processamento de variáveis numéricas - Padronização
"""

for n in nums_valores:
    dados[n] = MinMaxScaler().fit_transform(dados[n].values.reshape(len(dados), 1))

# Salvando novo dataset processado
dados.to_csv('./dados/dados_processados.csv', sep=',', encoding = 'utf-8', index = False)


"""
Separação de dados para Treino x Teste
"""

dados = pd.read_csv("C:/Projetos Pessoais/DataScience/analise_Previsao_Churn_teleComunicacoes/dados/dados_processados.csv")

'''Separação de dados para treino e teste'''
# Criando objeto para variável target
y = dados.Churn
# Criando objeto para variaveis de entrada
X = dados.drop('Churn', axis = 1)
# função train_test_split para separar os dados 
X_treino, X_teste, y_treino, y_teste = train_test_split(X, 
                                                        y, 
                                                        test_size = 0.30, 
                                                        random_state = 1234,
                                                        stratify = dados.Churn)

''' Balanceamento da classe alvo nos dados de treino'''
print(y_treino.value_counts()) # 0=3622 e 1=1308

over_sampler = SMOTE(k_neighbors = 2)
# Aplica o oversampling com dados de treino
x_res, y_res = over_sampler.fit_resample(X_treino, y_treino)

# Ajustando o nome do dataset de treino para X
X_treino = x_res
y_treino = y_res

print(y_res.value_counts()) # 0 = 3622 e 1 = 3622
# Variável 

for i in dados:
    valor_correlacao = dados[['Churn',i]].corr().loc['Churn',i]
    print(f"Correlacao de {i} com Churn: {valor_correlacao}")
