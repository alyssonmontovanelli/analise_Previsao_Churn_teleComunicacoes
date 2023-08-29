from utils import *
from pre_processamento import *
from ml_decisionTree import df_modelo_v2
from ml_knn import df_modelos_v3
from ml_naiveBayes import df_modelos_v6
from ml_regressionLog import df_modelo_v1
from ml_svm import df_modelos_v5
from ml_randomForest import df_modelos_v4

"""
Seleção de melhor modelo
"""

df_final = pd.concat([df_modelo_v1, df_modelo_v2, df_modelos_v3, df_modelos_v4, df_modelos_v5, df_modelos_v6], ignore_index = True)


print(df_final)
df_final.to_csv('C:/Projetos Pessoais/DataScience/analise_Previsao_Churn_teleComunicacoes/dados/df_comparativo_modelos', sep =',', index = False, encoding = 'utf-8')
 
# Resultado final 
'''
        Nome            Algoritmo  ROC_AUC Score  AUC Score  Acuracia
0  modelo_v1  Logistic Regression       0.737474   0.823714  0.749763
1  modelo_v2        Decision Tree       0.735860   0.805937  0.747393
2  modelo_v3                  KNN       0.730468   0.811358  0.698578
3  modelo_v4        Random Forest       0.694950   0.805813  0.761611
4  modelo_v5                  SVM       0.699501   0.737614  0.723223
5  modelo_v6          Naive Bayes       0.723204   0.795577  0.707109
'''


