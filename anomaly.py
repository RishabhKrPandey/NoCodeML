from pycaret.anomaly import setup, create_model, assign_model, pull, save_model
import pandas as pd

def train_anomaly(df):
    setup(df)
    setup_df = pull()
    iforest_model = create_model('iforest')
    results_df = assign_model(iforest_model)
    result_summary = pull()
    save_model(iforest_model, 'best_model')
    return setup_df, result_summary, results_df
