from utils import *
from pre_processamento import *

"""
------------------------------------------------------------------------------------------
Modelo 04 - Random Forest
------------------------------------------------------------------------------------------
"""

# Grid de hiperparametros
tuned_params_v2 = {
    'n_estimators': [50, 100, 200, 300],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
# Cria o modelo com RandomizedSearchCV para buscar a melhor combinação de hiperparametros
modelo_v4 = GridSearchCV(RandomForestClassifier(), tuned_params_v2, scoring='roc_auc', cv=5, n_jobs=-1)


# Treinamento do modelo
modelo_v4.fit(X_treino, y_treino)
print(modelo_v4.best_estimator_)
print(modelo_v4.best_params_)
# Previsões do modelo
y_pred_v4 = modelo_v4.predict(X_teste)

# Probabilidade
y_proba_v4 = modelo_v4.predict_proba(X_teste)[:,1]

# Matriz de confusão
# print(confusion_matrix(y_teste, y_pred_v4))

'''Métricas'''
# Curva ROC nos dados e previsões em teste
roc_auc_v2 = roc_auc_score(y_teste, y_pred_v4)

# AUC em teste
fpr_v2, tpr_v2, thresholds = roc_curve(y_teste, y_proba_v4)
auc_v2 = auc(fpr_v2, tpr_v2)

# Acurácia em teste
acuracia_v2 = accuracy_score(y_teste, y_pred_v4)


'''Recriando o modelo'''
modelo_v4 = RandomForestClassifier(n_estimators = 300,
                                  min_samples_split = 2,
                                  min_samples_leaf = 1)
modelo_v4.fit(X_treino, y_treino)


# DF com métricas
dict_modelo_v4 = {
    'Nome': ['modelo_v4'],
    'Algoritmo': ['Random Forest'],
    'ROC_AUC Score': [roc_auc_v2],
    'AUC Score': [auc_v2],
    'Acuracia':[acuracia_v2]
}


indices = np.argsort(-modelo_v4.feature_importances_)
# print("Vaiáveis mais importantes para o resultado do modelo")
# print(50 * '-')
# # for feature in X.columns[indices]:
# #     print(feature)

df_modelos_v4 = pd.DataFrame(dict_modelo_v4)
print(df_modelos_v4)

# Salvanndo novamente o modelo em disco 
with open('modelos/modelo_v4.pkl', 'wb') as pickle_file:
    joblib.dump(modelo_v4, 'modelos/modelo_v4.pkl')


'''
fidelidade
valor_total_pago
valor_mensal
contrato_Month-to-month
contrato_Two year
servico_internet_Fiber optic
suporte_tecnico_No
forma_pagamento_Electronic check
contrato_One year
suporte_tecnico_No internet service
suporte_tecnico_Yes
fatura_sem_papel_No
casado(a)_Yes
servico_internet_DSL
fatura_sem_papel_Yes
forma_pagamento_Bank transfer (automatic)
casado(a)_No
servico_internet_No
Streaming_TV_No internet service
possuiDependente_Yes
Streaming_TV_No
forma_pagamento_Credit card (automatic)
possuiDependente_No
idoso_No
Streaming_TV_Yes
forma_pagamento_Mailed check
idoso_Yes
'''