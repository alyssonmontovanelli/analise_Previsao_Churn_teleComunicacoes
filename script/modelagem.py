from utils import *

dados = pd.read_csv("C:/Projetos Pessoais/DataScience/analise_Previsao_Churn_teleComunicacoes/dados/dados_processados.csv")
print(dados.head())

'''Separação de dados para treino e teste'''
print(dados.shape)
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
print(len(x_res))
print(len(y_res))

print(y_res.value_counts()) # 0 = 3622 e 1 = 3622
# Variável 


