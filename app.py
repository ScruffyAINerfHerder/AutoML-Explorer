import streamlit as st
import pandas as pd
import os

#Import profiling capability
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report 

#ML libraries from PyCaret
from pycaret.classification import setup, compare_models, pull, save_model

#Create the sidebar
with st.sidebar:
    st.title("AutoML Explorer")
    choice = st.radio("Navigation", ["Upload", "Profiling", "AutoML", "Download"])
    st.image("Databot1.jpg")
    st.info("This app allows you to upload a dataset, profile it with pandas and analyze it with pycaret it is also build on Python and Streamlit ")

if os.path.exists("sourcedata.csv"):
    df = pd.read_csv("sourcedata.csv", index_col=None)

if choice == "Upload":
    st.title('Upload you data for modelling')
    file = st.file_uploader("Upload Dataset Here")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv("sourcedata.csv", index=None)
        st.dataframe(df)
    
if choice == "Profiling":
    st.title("Automated Data Explorer Report")
    profile_report = df.profile_report()
    st_profile_report(profile_report)

if choice == "AutoML":
    st.title("Machine Learning")
    target = st.selectbox("Select Your Target", df.columns)
    if st.button ("Train Model"):
        setup(df, target=target)
        setup_df = pull()
        st.info("ML Settings")
        st.dataframe(setup_df)
        best_model= compare_models()
        compare_df = pull()
        st.info("ML Model")
        st.dataframe(compare_df)
        best_model
        save_model(best_model, 'best_model')
    
if choice == "Download":
    st.title("Download the .pkl model")
    with open ('best_model.pkl', 'rb') as f:
        st.download_button("Download the Model", f, "trained_model.pkl")
    st.info("Once the trained model is downloaded, use Jupyter notebook to open and run")