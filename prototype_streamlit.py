# Step 1: Install necessary libraries
# Run the following command in your terminal:
# pip install streamlit matplotlib seaborn plotly

# Step 2: Import Libraries
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pandas as pd
import numpy as np
import joblib  # Assuming you used joblib to save your model
import plotly.graph_objects as go

# Step 3: Load Model and Data
model = joblib.load('./mdl_versions/PETR4_2024-09-12.joblib')
df = pd.read_csv('./databases/PETR4_FEAT_2024-09-12.csv')

# Step 4: Create Streamlit Layout
st.title("Strock Price Prediction App")
st.sidebar.header("Options")

# Step 5: Display Model Results
st.header("Model Predictions")
df['predicted'] = model.predict(df[model.feature_name_])

df.sort_values('date', inplace=True)
temp = df[df['date'] > '2024-07-01']

# Create a Plotly figure
fig = go.Figure()
# Add observed data
fig.add_trace(go.Scatter(x=temp['date'], y=temp['target'], mode='lines', name='Observed', line=dict(color='orange')))
# Add predicted data
fig.add_trace(go.Scatter(x=temp['date'], y=temp['predicted'], mode='lines', name='Predicted', line=dict(color='orange', dash='dash')))
# Update layout
fig.update_layout(
    title='Stock Close Price per Day',
    xaxis_title='Date',
    yaxis_title='Close Price',
    legend=dict(x=0, y=1)
)

st.plotly_chart(fig)