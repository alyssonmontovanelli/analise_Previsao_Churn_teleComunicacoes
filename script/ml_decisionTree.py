from utils import *
from pre_processamento import *


"""
------------------------------------------------------------------------------------------
Modelo 02 - Árvore de Decisão 
------------------------------------------------------------------------------------------
"""
# Hiperparâmetros 
tuned_params_v2 = {
    'min_samples_split': [2, 3, 4, 5, 7],
    'min_samples_leaf': [1, 2, 3, 4, 6],
    'max_depth': [2, 3, 4, 5, 6, 7]
}

modelo_v2 = RandomizedSearchCV(DecisionTreeClassifier(),
                              tuned_params_v2,
                              n_iter = 15,
                              scoring = 'roc_auc',
                              n_jobs = -1)

modelo_v2.fit(X_treino, y_treino)
modelo_v2.best_estimator_
# Melhores hiperparams
print(modelo_v2.best_estimator_)

# Previsões
y_pred_v2 = modelo_v2.predict(X_teste)
print(y_pred_v2[:15])


# Previsoes proba
y_pred_proba_v2 = modelo_v2.predict_proba(X_teste)[:,1]
print(modelo_v2.predict_proba(X_teste)[:15])

# Matriz de confusão
print(confusion_matrix(y_teste, y_pred_v2))

# ROC AUC score
roc_auc_v2 = roc_auc_score(y_teste, y_pred_v2)
print(roc_auc_v2)

# Curva ROC
fpr_v2, tpr_v2, thresholds = roc_curve(y_teste, y_pred_proba_v2)
auc_v2 = auc(fpr_v2, tpr_v2)
print(auc_v2)

# acurácia
acuracia_v2 = accuracy_score(y_teste, y_pred_v2)
print(acuracia_v2)


''' Recriação do modelo - hiperparams otimizados'''
modelo_v2 = DecisionTreeClassifier(max_depth=6, min_samples_leaf=3, min_samples_split=5)
modelo_v2.fit(X_treino, y_treino)

# Variáceis mais importantes
indices = np.argsort(-modelo_v2.feature_importances_)
print("Vaiáveis mais importantes para o resultado do modelo")
print(50 * '-')
for feature in X.columns[indices]:
    print(feature)



''' Maior relevancia
contrato_Month-to-month
valor_total_pago
valor_mensal
servico_internet_Fiber optic
contrato_One year
contrato_Two year
Streaming_TV_No internet service
fatura_sem_papel_No
suporte_tecnico_Yes
forma_pagamento_Bank transfer (automatic)
Streaming_TV_No
possuiDependente_Yes
fidelidade_2 a 3 anos
possuiDependente_No
fidelidade_+5 anos
forma_pagamento_Credit card (automatic)
fidelidade_4 a 5 anos
idoso_Yes
forma_pagamento_Mailed check
casado(a)_Yes
Streaming_TV_Yes
fidelidade_-1 ano
suporte_tecnico_No
forma_pagamento_Electronic check
servico_internet_No
servico_internet_DSL
casado(a)_No
fatura_sem_papel_Yes
idoso_No
fidelidade_3 a 4 anos
suporte_tecnico_No internet service
fidelidade_1 a 2 anos
'''

with open('modelos/modelo_v2.pkl', 'wb') as pickle_file:
    joblib.dump(modelo_v2, 'modelos/modelo_v2.pkl')

dict_modelo_v2 = {'Nome': ['modelo_v1'], 
                  'Algoritmo': ['Logistic Regression'], 
                  'ROC_AUC Score': [roc_auc_v2],
                  'AUC Score': [auc_v2],
                  'Acurácia': [acuracia_v2]}
df_modelo_v1 = pd.DataFrame(dict_modelo_v2)
print(df_modelo_v1)