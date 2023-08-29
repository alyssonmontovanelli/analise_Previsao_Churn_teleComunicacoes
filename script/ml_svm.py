from utils import *
from pre_processamento import *

"""
------------------------------------------------------------------------------------------
Modelo 05 - SVM - Support Vector Machines
------------------------------------------------------------------------------------------
"""

# Seleção de hiper paramtros
def svc_param_selection(x, y, nfolds):
    Cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1]
    param_grid = {'C': Cs, 'gamma': gammas}
    grid_search = GridSearchCV(SVC(kernel = 'rbf'), param_grid, cv = nfolds)
    grid_search.fit(X_treino, y_treino)
    grid_search.best_params_
    return grid_search.best_params_

print(svc_param_selection(X_treino, y_treino, 5))


'''Modelo'''
modelo_v5 = SVC(C = 10, gamma = 1, probability = True)

modelo_v5.fit(X_treino, y_treino)

# Previsões
y_pred_v5 = modelo_v5.predict(X_teste)
print(y_pred_v5)

print(confusion_matrix(y_teste, y_pred_v5))

# Previsões - proba
y_proba_v5 = modelo_v5.predict_proba(X_teste)[:,1]
print(modelo_v5.predict_proba(X_teste))

roc_auc_v5 = roc_auc_score(y_teste, y_pred_v5)
print(roc_auc_v5)

# Curva ROC
fpr_v5, tpr_v5, thresholds = roc_curve(y_teste, y_proba_v5)
auc_v5 = auc(fpr_v5, tpr_v5)
print(auc_v5)

# acuracia
acuracia_v5 = accuracy_score(y_teste, y_pred_v5)
print(acuracia_v5)

dict_modelo_v5 = {
    'Nome': ['modelo_v5'],
    'Algoritmo': ['SVM'],
    'ROC_AUC Score': [roc_auc_v5],
    'AUC Score': [auc_v5],
    'Acuracia':[acuracia_v5]
}
df_modelos_v5 = pd.DataFrame(dict_modelo_v5)
print(df_modelos_v5)

with open('modelos/modelo_v5.pkl', 'wb') as pickle_file:
    joblib.dump(modelo_v5, 'modelos/modelo_v5.pkl')