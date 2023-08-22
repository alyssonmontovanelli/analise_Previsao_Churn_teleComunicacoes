from utils import *
from pre_processamento import *

"""
------------------------------------------------------------------------------------------
Modelo 01 - Regressão Logística (Benchmark)
------------------------------------------------------------------------------------------
"""
# lista de hiperparâmetros
parametros_v1 = {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000], 
                   'penalty': ['l2']}

modelo_v1 = GridSearchCV(LogisticRegression(solver='liblinear', max_iter=1000), 
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
# print(y_pred_proba_v1[:10])
print(modelo_v1.predict_proba(X_teste)[:10])

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

# Construção do modelo sem o GridSearchVC
modelo_v1 = LogisticRegression(C = 1000)
modelo_v1.fit(X_treino, y_treino)

# Variáveis mais relevantes
indices = np.argsort(-abs(modelo_v1.coef_[0,:]))
print('variaveis mais relevantes')
for feature in X.columns[indices]:
    print(feature)
    
#Salvar em disco
with open('modelos/modelo_v1.pk1', 'wb') as pickle_file:
    joblib.dump(modelo_v1, 'modelos/modelo_v1.pk1')

dict_modelo_v1 = {'Nome': ['modelo_v1'], 
                  'Algoritmo': ['Logistic Regression'], 
                  'ROC_AUC Score': [roc_auc_v1],
                  'AUC Score': [auc_v1],
                  'Acurácia': [acuracia_v1]}
df_modelo_v1 = pd.DataFrame(dict_modelo_v1)
print(df_modelo_v1)

