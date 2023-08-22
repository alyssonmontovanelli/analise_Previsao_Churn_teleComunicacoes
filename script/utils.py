import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
import sklearn

'''
Imports para modelagem
'''
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix
from sklearn.metrics import accuracy_score

'''
Import para balanceamento de classe
'''
import imblearn
from imblearn.over_sampling import SMOTE


'''
Mapeamento idoso
'''
map_idoso = {
    0: 'No',
    1: 'Yes'
}

'''
Mapeamento fidelidade
'''
def map_fidelidade(mes):
    if 0 <= mes <=12:
        return '-1 ano'
    elif 13 <= mes <= 24:
        return '1 a 2 anos'
    elif 25 <= mes <= 36:
        return '2 a 3 anos'
    elif 37 <= mes <= 48:
        return '3 a 4 anos'
    elif 49 <= mes <= 60:
        return '4 a 5 anos'
    elif mes > 60:
        return '+5 anos'


cats = ['fidelidade', 'idoso', 'casado(a)', 'possuiDependente', 'servico_internet',
        'suporte_tecnico', 'Streaming_TV', 'contrato','fatura_sem_papel',
        'forma_pagamento']


variaveis_processadas = ['fidelidade_+5 anos', 'fidelidade_-1 ano',
       'fidelidade_1 a 2 anos', 'fidelidade_2 a 3 anos',
       'fidelidade_3 a 4 anos', 'fidelidade_4 a 5 anos', 'idoso_No',
       'idoso_Yes', 'casado(a)_No', 'casado(a)_Yes', 'possuiDependente_No',
       'possuiDependente_Yes', 'servico_internet_DSL',
       'servico_internet_Fiber optic', 'servico_internet_No',
       'suporte_tecnico_No', 'suporte_tecnico_No internet service',
       'suporte_tecnico_Yes', 'Streaming_TV_No',
       'Streaming_TV_No internet service', 'Streaming_TV_Yes',
       'contrato_Month-to-month', 'contrato_One year', 'contrato_Two year',
       'fatura_sem_papel_No', 'fatura_sem_papel_Yes',
       'forma_pagamento_Bank transfer (automatic)',
       'forma_pagamento_Credit card (automatic)',
       'forma_pagamento_Electronic check', 'forma_pagamento_Mailed check']

nums_valores = ['valor_mensal', 'valor_total_pago']