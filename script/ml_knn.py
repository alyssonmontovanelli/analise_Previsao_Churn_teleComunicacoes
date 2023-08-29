from utils import *
from pre_processamento import *

"""
------------------------------------------------------------------------------------------
Modelo 03 - KNN
------------------------------------------------------------------------------------------
"""

# Lista de possíveis valores de K
vizinhos = list(range(1, 20, 2))
# Lista para os scores
cv_scores = []
# Validade cruzada para determinar o melhor valor de k

for k in vizinhos:
    knn = KNeighborsClassifier(n_neighbors = k) # Cria o modelo para cada valor de K
    scores = cross_val_score(knn, X_treino, y_treino, cv = 5, scoring = 'accuracy') # Validação cruzada
    cv_scores.append(scores.mean()) #calculo score
# Ajustando o erro de classificação
erro = [1 - x for x in cv_scores]
# Determinando o melhor valor de k ( com menor erro )
optimal_k = vizinhos[erro.index(min(erro))]
# print(f'O valor ideal de k: {optimal_k}')

# Criação do modelo
modelo_v3 = KNeighborsClassifier(n_neighbors = 101)
modelo_v3.fit(X_treino, y_treino)

# Previsões
y_pred_v3 = modelo_v3.predict(X_teste)
# print(y_pred_v3[:10])

# Matriz de confusão
# print(confusion_matrix(y_teste, y_pred_v3))

# Probabilidade
y_proba_v3 = modelo_v3.predict_proba(X_teste)[:,1]
# print(modelo_v3.predict_proba(X_teste)[:10])

# ROC_AUC em teste
roc_auc_v3 = roc_auc_score(y_teste, y_pred_v3)

# Calculando AUC em Teste
fpr_v3, tpr_v3, thresholds = roc_curve(y_teste, y_proba_v3)
auc_v3 = auc(fpr_v3, tpr_v3)

# Calculando a acurácia
acuracia_v3 = accuracy_score(y_teste, y_pred_v3)

# DF com métricas
dict_modelo_v3 = {
    'Nome': ['modelo_v3'],
    'Algoritmo': ['KNN'],
    'ROC_AUC Score': [roc_auc_v3],
    'AUC Score': [auc_v3],
    'Acuracia':[acuracia_v3]
}

df_modelos_v3 = pd.DataFrame(dict_modelo_v3)
print(df_modelos_v3)

# Salvanndo novamente o modelo em disco 
with open('modelos/modelo_v3.pkl', 'wb') as pickle_file:
    joblib.dump(modelo_v3, 'modelos/modelo_v3.pkl')