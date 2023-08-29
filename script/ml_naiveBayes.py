from utils import *
from pre_processamento import *

"""
------------------------------------------------------------------------------------------
Modelo 06 - Naive Bayes
------------------------------------------------------------------------------------------
"""


param_grid = {
    'alpha': [0.1, 0.5, 1.0],  # Parâmetro de suavização Laplaciana
    'fit_prior': [True, False]  # Aprender a priori das classes ou não
}

# Crie uma instância do modelo Naive Bayes Multinomial
modelo_nb = MultinomialNB()

# Crie um objeto GridSearchCV
grid_search = GridSearchCV(modelo_nb, 
                           param_grid, 
                           cv=5, 
                           scoring='accuracy')

# Ajuste o objeto GridSearchCV aos dados de treinamento
grid_search.fit(X_treino, y_treino)

# Imprima os melhores parâmetros encontrados
print("Melhores Parâmetros:", grid_search.best_params_)


modelo_v6 = MultinomialNB(alpha= 0.1, fit_prior= True)
modelo_v6.fit(X_treino, y_treino)

# Previsões
y_pred_v6 = modelo_v6.predict(X_teste)
print(y_pred_v6)

print(confusion_matrix(y_teste, y_pred_v6))

# Previsões - proba
y_proba_v6 = modelo_v6.predict_proba(X_teste)[:,1]
print(modelo_v6.predict_proba(X_teste))

roc_auc_v6 = roc_auc_score(y_teste, y_pred_v6)
print(roc_auc_v6)

# Curva ROC
fpr_v6, tpr_v6, thresholds = roc_curve(y_teste, y_proba_v6)
auc_v6 = auc(fpr_v6, tpr_v6)
print(auc_v6)

# acuracia
acuracia_v6 = accuracy_score(y_teste, y_pred_v6)
print(acuracia_v6)

dict_modelo_v6 = {
    'Nome': ['modelo_v6'],
    'Algoritmo': ['Naive Bayes'],
    'ROC_AUC Score': [roc_auc_v6],
    'AUC Score': [auc_v6],
    'Acuracia':[acuracia_v6]
}
df_modelos_v6 = pd.DataFrame(dict_modelo_v6)
print(df_modelos_v6)

with open('modelos/modelo_v6.pkl', 'wb') as pickle_file:
    joblib.dump(modelo_v6, 'modelos/modelo_v6.pkl')
