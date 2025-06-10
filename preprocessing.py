import pycaret
import pandas as pd


from pycaret.classification import setup as cls_setup
from pycaret.regression import setup as reg_setup
from pycaret.clustering import setup as clu_setup
from pycaret.anomaly import setup as ano_setup
from pycaret.time_series import setup as ts_setup
from pycaret.datasets import get_data

def preprocess_data(df, task_type, target=None):
    if task_type == 'regression':
        reg_setup(df, target=target, preprocess=True)
    elif task_type == 'classification':
        cls_setup(df, target=target, preprocess=True)
    elif task_type == 'clustering':
        clu_setup(df, preprocess=True)
    elif task_type == 'anomaly':
        ano_setup(df, preprocess=True)
    elif task_type == 'time_series':
        ts_setup(df, target=target, session_id=123, fold=3)
    return df

