from utils import *

dados = pd.read_csv("C:/Projetos Pessoais/DataScience/analise_Previsao_Churn_teleComunicacoes/dados/dados_processados.csv")
print(dados.head())

'''Separação de dados para treino e teste'''
print(dados.shape)
# Criando objeto para variável target
y = dados.Churn
# Criando objeto para variaveis de entrada
X = dados.drop('Churn', axis = 1)
