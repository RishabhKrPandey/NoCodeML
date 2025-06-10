import streamlit as st
import pandas as pd
import os

from preprocessing import preprocess_data
from regression import train_regression
from classification import train_classification
from clustering import train_clustering
from anomaly import train_anomaly
from time_series import train_time_series

from pycaret.regression import predict_model
from pycaret.classification import predict_model as cls_predict
from pycaret.clustering import predict_model as clu_predict
from pycaret.anomaly import predict_model as ano_predict
from pycaret.time_series import predict_model as ts_predict

from pycaret.regression import load_model as reg_load
from pycaret.classification import load_model as cls_load
from pycaret.clustering import load_model as clu_load
from pycaret.anomaly import load_model as ano_load
from pycaret.time_series import load_model as ts_load




# Load existing dataset
df = None
if os.path.exists('dataset.csv'):
    df = pd.read_csv('dataset.csv', index_col=None)

# Sidebar setup
st.sidebar.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
st.sidebar.title("NoCodeML")
choice = st.sidebar.radio("Navigation", ["Upload", "Profiling", "Preprocessing", "Modeling", "Predict", "Download"])

# Upload
if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload CSV File")
    if file:
        df = pd.read_csv(file)
        df.to_csv('dataset.csv', index=False)
        st.dataframe(df)

# Profiling
elif choice == "Profiling":
    if 'df' in locals():
        st.title("Basic Data Profiling Report")

        st.subheader("1. Dataset Preview")
        st.dataframe(df.head())

        st.subheader("2. Dataset Shape")
        st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

        st.subheader("3. Column Types")
        st.write(df.dtypes)

        st.subheader("4. Summary Statistics (Numerical)")
        st.dataframe(df.describe())

        st.subheader("5. Missing Values")
        st.dataframe(df.isnull().sum().reset_index().rename(columns={0: 'Missing Count', 'index': 'Column'}))

        st.subheader("6. Unique Values per Column")
        st.dataframe(df.nunique().reset_index().rename(columns={0: 'Unique Count', 'index': 'Column'}))

        st.subheader("7. Correlation Matrix")
        corr = df.select_dtypes(include=['float64', 'int64']).corr()
        st.dataframe(corr)

        st.subheader("8. Value Counts for Categorical Columns")
        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            st.markdown(f"**{col}**")
            st.dataframe(df[col].value_counts())
    else:
        st.warning("Please upload a dataset first.")


# Preprocessing
elif choice == "Preprocessing":
    if df is not None:
        st.title("Preprocess Data")
        task = st.selectbox("Choose ML Task", ["regression", "classification", "clustering", "anomaly", "time_series"])
        target = None
        if task in ["regression", "classification", "time_series"]:
            target = st.selectbox("Select Target Column", df.columns)
        if st.button("Preprocess"):
            processed_df = preprocess_data(df.copy(), task, target)
            processed_df.to_csv("processed_dataset.csv", index=False)
            st.success("Preprocessing Completed!")
            st.dataframe(processed_df)


    else:
        st.warning("Please upload a dataset first.")

# Modeling
elif choice == "Modeling":
    if df is not None:
        st.title("Train ML Model")
        task = st.selectbox("Choose ML Task", ["regression", "classification", "clustering", "anomaly", "time_series"])
        target = None
        if task in ["regression", "classification", "time_series"]:
            target = st.selectbox("Select Target Column", df.columns)

        if st.button("Train Model"):
            if task == "regression":
                setup_df, compare_df = train_regression(df, target)
            elif task == "classification":
                setup_df, compare_df = train_classification(df, target)
            elif task == "clustering":
                setup_df, compare_df = train_clustering(df)
            elif task == "anomaly":
                setup_df, compare_df = train_anomaly(df)
            elif task == "time_series":
                setup_df, compare_df = train_time_series(df, target)

            st.subheader("Setup Summary")
            st.dataframe(setup_df)
            st.subheader("Model Comparison")
            st.dataframe(compare_df)
    else:
        st.warning("Please upload a dataset first.")

# Predict
elif choice == "Predict":
    st.title("Real-Time Prediction")
    input_data = st.text_area("Enter data row (comma-separated):")
    model_type = st.selectbox("Select Model Type",
                              ["regression", "classification", "clustering", "anomaly", "time_series"])

    if st.button("Predict"):
        try:
            # Automatically detect columns (excluding target)
            input_cols = df.drop(columns=[df.columns[-1]]).columns.tolist()
            values = input_data.strip().split(",")

            if len(values) != len(input_cols):
                st.error(f"Expected {len(input_cols)} values, but got {len(values)}.")
            else:
                # Create a DataFrame row with the correct column names and original string values
                row = pd.DataFrame([values], columns=input_cols)

                # Load appropriate model and predict
                if model_type == "regression":
                    model = reg_load("best_model")
                    prediction = predict_model(model, data=row)
                elif model_type == "classification":
                    model = cls_load("best_model")
                    prediction = cls_predict(model, data=row)
                elif model_type == "clustering":
                    model = clu_load("best_model")
                    prediction = clu_predict(model, data=row)
                elif model_type == "anomaly":
                    model = ano_load("best_model")
                    prediction = ano_predict(model, data=row)
                elif model_type == "time_series":
                    model = ts_load("best_model")
                    prediction = ts_predict(model, data=row)

                st.subheader("Prediction Result")
                st.write(prediction)

        except Exception as e:
            st.error(f"Prediction failed: {e}")


# Download
elif choice == "Download":
    if os.path.exists("best_model.pkl"):
        with open("best_model.pkl", "rb") as f:
            st.download_button("Download Best Model", f, "best_model.pkl")
    else:
        st.warning("No trained model found.")
