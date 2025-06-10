from pycaret.clustering import setup, create_model, assign_model, pull, save_model
import pandas as pd

def train_clustering(df):
    setup(df)
    setup_df = pull()
    kmeans_model = create_model('kmeans')
    clustered_df = assign_model(kmeans_model)
    cluster_results = pull()
    save_model(kmeans_model, 'best_model')
    return setup_df, cluster_results, clustered_df
