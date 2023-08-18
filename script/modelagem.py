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
# Ajustando o nome do dataset de treino para X
X_treino = x_res
y_treino = y_res

print(y_res.value_counts()) # 0 = 3622 e 1 = 3622
# Variável 


"""
------------------------------------------------------------------------------------------
Modelo 01 - Regressão Logística (Benchmark)
------------------------------------------------------------------------------------------
"""
# lista de hiperparâmetros
parametros_v1 = {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000], 
                   'penalty': ['l2']}

modelo_v1 = GridSearchCV(LogisticRegression(solver='liblinear'), 
                         parametros_v1, 
                         scoring = 'roc_auc', 
                         n_jobs = -1)

# Treinamento do modelo
modelo_v1.fit(X_treino, y_treino)

# Buscando melhor hiper param
modelo_v1.best_estimator_
print(modelo_v1.best_estimator_)

# Previsões com dados de teste e métricas
y_pred_v1 = modelo_v1.predict(X_teste)
print(y_pred_v1[:10])

# previsao no formato de probabilidade
y_pred_proba_v1 = modelo_v1.predict_proba(X_teste)[:,1]
print(y_pred_proba_v1[:10])


print('Para o ponto de dado {}, classe real = {}, classe prevista = {}, probabilidade prevista = {}'.
      format(16, y_teste.iloc[16], y_pred_v1[16], modelo_v1.predict_proba(X_teste)[16]))

# Matriz de confusão
confusion_matrix(y_teste, y_pred_v1)
tn, fp, fn, tp = confusion_matrix(y_teste, y_pred_v1).ravel()
print(tn, fp, fn, tp)

# Calcula a métrica global AUC (Area Under The Curve) com dados reais e previsões em teste
roc_auc_v1 = roc_auc_score(y_teste, y_pred_v1)
print(roc_auc_v1)

# Calcula a curva ROC com dados e previsões em teste
fpr_v1, tpr_v1, thresholds = roc_curve(y_teste, y_pred_proba_v1)
# AUC em teste
auc_v1 = auc(fpr_v1, tpr_v1)
print(auc_v1)

# Acurácia em teste
acuracia_v1 = accuracy_score(y_teste, y_pred_v1)
print(acuracia_v1)