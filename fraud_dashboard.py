import streamlit as sl
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as mpl
from sklearn.ensemble import RandomForestClassifier
import joblib 

sl.title("Fraud Detection Dashboard")
sl.markdown("Upload a dataset to explore fraud patterns and view simulated model predictions.")

uploaded_file = sl.file_uploader(" Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    sl.subheader(" Data Preview")
    sl.write(df.head())

    if 'Class' in df.columns:
        X = df.drop(['Class', 'Time'], axis=1, errors='ignore')
    else:
        X = df

    if 'Class' in df.columns:
        fraud_rate = df['Class'].value_counts(normalize=True).get(1, 0) * 100
        sl.metric("Fraud Rate", f"{fraud_rate:.2f}%")

        # Visual chart
        sl.subheader("Class Distribution")
        fig, ax = mpl.subplots()
        sb.countplot(data=df, x='Class', palette='Set2', ax=ax)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Not Fraud (0)', 'Fraud (1)'])
        ax.set_ylabel("Count")
        sl.pyplot(fig)
    else:
        sl.error("This dataset must include a 'Class' column to calculate fraud stats.")

    
    model = joblib.load("fraud_model.pkl")
    predictions = model.predict(X.fillna(0))
    df['Predicted Fraud'] = predictions

    sl.subheader(" Predicted Fraudulent Transactions")
    sl.dataframe(df[df['Predicted Fraud'] == 1].head(10))

   




   



    


   






   
