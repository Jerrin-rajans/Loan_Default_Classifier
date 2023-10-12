# -*- coding: utf-8 -*-
# Streamlit
import streamlit as st 
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import pickle
import numpy as np

st.title('Loan Default Prediction App')
enc=OrdinalEncoder()
test_cols = ['Client_Occupation','Social_Circle_Default']
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    df1 = pd.read_csv(uploaded_file)
    drop_cols=['ID','Application_Process_Day','Application_Process_Hour']
    df1.drop(columns=drop_cols,axis=1,inplace=True)
    df1['Score_Source_3']=df1['Score_Source_3'].str.replace('&','').replace('',np.nan)
    df1['Score_Source_3']=df1['Score_Source_3'].str.replace('#','').replace('',np.nan)
    df1['Score_Source_3']=df1['Score_Source_3'].astype(float)
    df1['Client_Income']=df1['Client_Income'].str.replace('$','').replace('',np.nan)
    df1['Client_Income']=df1['Client_Income'].astype(float)
    df1['Credit_Amount']=df1['Credit_Amount'].str.replace('$','').replace('',np.nan)
    df1['Credit_Amount']=df1['Credit_Amount'].astype(float)
    df1['Loan_Annuity']=df1['Loan_Annuity'].str.replace('#VALUE!','').replace('',np.nan)
    df1['Loan_Annuity']=df1['Loan_Annuity'].str.replace('$','').replace('',np.nan)
    df1['Loan_Annuity']=df1['Loan_Annuity'].astype(float)
    df1.drop(test_cols,axis = 1, inplace=True)
    df1.dropna(inplace=True)
    for col in df1:
        if df1[col].dtype in [object,'category']:
            df1[col]=enc.fit_transform(df1[col].astype(str).values.reshape(-1,1))
        else:
            continue
    # Imputing test data below
    for col in df1:
        if df1[col].dtype in [int,float]:
            mean_val = df1[col].mean()
            df1[col].fillna(mean_val,inplace=True)
    st.text('Model used is Extra Tree Regressor')
    pickled_model = pickle.load(open('model.pkl', 'rb'))
    result = pickled_model.predict(df1)
    df1['Default_Predicted'] = result
    st.dataframe(df1)
    @st.cache_data
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')
    csv = convert_df(df1)
    st.download_button("Press to Download",csv,"file.csv","text/csv",key='download-csv')
    
    