from utils import *
from pre_processamento import *

"""
------------------------------------------------------------------------------------------
Modelo 04 - Random Forest
------------------------------------------------------------------------------------------
"""

# Grid de hiperparametros
tuned_params_v2 = {
    'n_estimators': [100, 200, 300, 400, 500, 600],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 6]
}
# Cria o modelo com RandomizedSearchCV para buscar a melhor combinação de hiperparametros
modelo_v4 = RandomizedSearchCV(RandomForestClassifier(),
                              tuned_params_v2,
                              n_iter = 15,
                              scoring = 'roc_auc',
                              n_jobs = -1)
# Treinamento do modelo
modelo_v4.fit(X_treino, y_treino)

print(modelo_v4.best_params_)
print(modelo_v4.best_estimator_)
print(modelo_v4.best_index_)

# Previsões do modelo
y_pred_v4 = modelo_v4.predict(X_teste)
print(y_pred_v4)

# Probabilidade
y_proba_v4 = modelo_v4.predict_proba(X_teste)[:,1]
print(modelo_v4.predict_proba(X_teste))

# Matriz de confusão
print(confusion_matrix(y_teste, y_pred_v4))

'''Métricas'''
# Curva ROC nos dados e previsões em teste
roc_auc_v2 = roc_auc_score(y_teste, y_pred_v4)
print(roc_auc_v2)

# AUC em teste
fpr_v2, tpr_v2, thresholds = roc_curve(y_teste, y_proba_v4)
auc_v2 = auc(fpr_v2, tpr_v2)
print(auc_v2)

# Acurácia em teste
acuracia_v2 = accuracy_score(y_teste, y_pred_v4)
print(acuracia_v2)


'''Recriando o modelo'''
modelo_v4 = RandomForestClassifier(n_estimators = 100,
                                  min_samples_split = 5,
                                  min_samples_leaf = 2)
modelo_v4.fit(X_treino, y_treino)


# DF com métricas
dict_modelo_v4 = {
    'Nome': ['modelo_v4'],
    'Algoritmo': ['Random Forest'],
    'ROC_AUC Score': [roc_auc_v2],
    'AUC Score': [auc_v2],
    'Acurácia':[acuracia_v2]
}

df_modelos_v4 = pd.DataFrame(dict_modelo_v4)
print(df_modelos_v4.head())

# Salvanndo novamente o modelo em disco 
with open('modelos/modelo_v4.pkl', 'wb') as pickle_file:
    joblib.dump(modelo_v4, 'modelos/modelo_v4.pkl')