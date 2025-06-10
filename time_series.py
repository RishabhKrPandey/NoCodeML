from pycaret.time_series import setup, compare_models, pull, save_model
import pandas as pd

def train_time_series(df, time_column):
    df[time_column] = pd.to_datetime(df[time_column])
    df.set_index(time_column, inplace=True)
    setup(data=df, target=df.columns[0], session_id=123)
    setup_df = pull()
    best_model = compare_models()
    compare_df = pull()
    save_model(best_model, 'best_model')
    return setup_df, compare_df
