
import numpy as np
import pandas as pd
import lightgbm
import gc
import time
from contextlib import contextmanager
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from helpers.data_prep import *
from helpers.eda import *


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)




def installments_payments(num_rows=None, nan_as_category=True):
    df = pd.read_csv(r'VBO_Proje/Datasets/installments_payments.csv', nrows=num_rows)
    cat_cols, num_cols, cat_but_car = grab_col_names(df)
    df, cat_cols = one_hot_encoder(df, categorical_cols=True)  ### neden cat var?

    #########################
    # Feature Engineering
    #########################
    # Her kredi taksidi ödemesinde ödediği miktarla aslı arasındaki fark ve bunun yüzdesi
    df['PAYMENT_PERC'] = df['AMT_PAYMENT'] / df['AMT_INSTALMENT']
    df['PAYMENT_DIFF'] = df['AMT_INSTALMENT'] - df['AMT_PAYMENT']

    # Vadesi geçmiş günler ve vadesinden önceki günler -- sadece pozitif değerler alınır
    df['DPD'] = df['DAYS_ENTRY_PAYMENT'] - df['DAYS_INSTALMENT']
    df['DBD'] = df['DAYS_INSTALMENT'] - df['DAYS_ENTRY_PAYMENT']
    df['DPD'] = df['DPD'].apply(lambda x: x if x > 0 else 0)
    df['DBD'] = df['DBD'].apply(lambda x: x if x > 0 else 0)

    # Her bir taksit ödemesinin gec olup olmama durumu 1: gec ödedi 0: erken ödemeyi temsil eder
    df['NEW_DAYS_PAID_EARLIER'] = df['DAYS_INSTALMENT'] - df['DAYS_ENTRY_PAYMENT']
    df['NEW_NUM_PAID_LATER'] = df['NEW_DAYS_PAID_EARLIER'].map(lambda x: 1 if x < 0 else 0)

    # Numeric Features
    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum'],
        'DBD': ['max', 'mean', 'sum'],
        'PAYMENT_PERC': ['max', 'mean', 'sum', 'var'],
        'PAYMENT_DIFF': ['max', 'mean', 'sum', 'var'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']
    }

    # Categorical Features
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    ins_agg = df.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
    # Count installments accounts
    ins_agg['INSTAL_COUNT'] = df.groupby('SK_ID_CURR').size()
    del df
    gc.collect()
    # print(ins_agg.columns.tolist())
    return ins_agg


installments_payments()