import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
import sklearn
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler


'''
Mapeamento idoso
'''
map_idoso = {
    0: 'No',
    1: 'Yes'
}

cats = ['idoso', 'casado(a)', 'possuiDependente', 'servico_telefonico', 'servico_internet',
        'protecao_dispositivo', 'suporte_tecnico', 'Streaming_TV', 'contrato','fatura_sem_papel',
        'forma_pagamento']


variaveis_processadas = ['idoso_No',
       'idoso_Yes', 'casado(a)_No', 'casado(a)_Yes', 'possuiDependente_No',
       'possuiDependente_Yes', 'servico_telefonico_No',
       'servico_telefonico_Yes', 'servico_internet_DSL',
       'servico_internet_Fiber optic', 'servico_internet_No',
       'protecao_dispositivo_No', 'protecao_dispositivo_No internet service',
       'protecao_dispositivo_Yes', 'suporte_tecnico_No',
       'suporte_tecnico_No internet service', 'suporte_tecnico_Yes',
       'Streaming_TV_No', 'Streaming_TV_No internet service',
       'Streaming_TV_Yes', 'contrato_Month-to-month', 'contrato_One year',
       'contrato_Two year', 'fatura_sem_papel_No', 'fatura_sem_papel_Yes',
       'forma_pagamento_Bank transfer (automatic)',
       'forma_pagamento_Credit card (automatic)',
       'forma_pagamento_Electronic check', 'forma_pagamento_Mailed check']

nums_valores = ['fidelidade', 'valor_mensal', 'valor_total_pago']