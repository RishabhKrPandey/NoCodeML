from pycaret.classification import setup, compare_models, pull, save_model
import pandas as pd

def train_classification(df, target):
    setup(df, target=target)
    setup_df = pull()
    best_model = compare_models()
    compare_df = pull()
    save_model(best_model, 'best_model')
    return setup_df, compare_df
