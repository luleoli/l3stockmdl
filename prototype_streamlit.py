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
from sklearn.metrics import r2_score, mean_absolute_error
import lightgbm as lgb
import shap
import streamlit.components.v1 as components

# Step 3: Load Model and Data
model = joblib.load('./mdl_versions/PETR4_2024-10-25.joblib')
df = pd.read_csv('./PETR4_FEAT_2024-10-25.csv')
df['predicted'] = model.predict(df[model.feature_name_])
df.sort_values('date', inplace=True)
# Filter the last 30 days
df['date'] = pd.to_datetime(df['date'])
temp = df[df['date'] >= (df['date'].max() - pd.Timedelta(days=30))]


predicted = temp['predicted'].values[-5:-1]
target = temp['target'].values[-5:-1]

# Calculate additional regression metrics

r2 = r2_score(predicted, target)
mae = mean_absolute_error(target, predicted)
mape = np.mean(np.abs((target - predicted) / target)) * 100


# Step 4: Create Streamlit Layout
st.title("Stock Prediction Model")
st.write("Based on the data, the model predicts the stock close price per day.")

# Step 5: Display Model Results
st.header("Predictions")


col1, col2 = st.columns([0.7, 0.3], vertical_alignment='center')

with col1:
    # Create a Plotly figure
    fig = go.Figure()
    # Add observed data
    fig.add_trace(go.Scatter(x=temp['date'], y=temp['predicted'], name='Predicted', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=temp['date'], y=temp['target'], mode='lines', name='Observed', line=dict(color='orange', width=1, dash='dash')))

    # Add predicted datamode='lines', name='Predicted', line=dict(color='orange', dash='dash')))
    # Update layout
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Close Price',
        legend=dict(x=0, y=1)
    )

    st.plotly_chart(fig)


with col2:
    # Display model information
    
    st.write('<div style="text-align: center"> Last Price: </div>', unsafe_allow_html=True)
    st.write(f"<h1 style='text-align: center'>{df['close'].values[-1]}</h1>", unsafe_allow_html=True)
    st.write('<div style="text-align: center"> Predicted Price: </div>', unsafe_allow_html=True)
    st.write(f"<h1 style='text-align: center'>{round(df['predicted'].values[-1],2)}</h1>", unsafe_allow_html=True)



col3, col4 = st.columns([0.7, 0.3], vertical_alignment='center')

with col3:
        # Step 5: Display Model Results
    st.header("Results")
    results_df = pd.DataFrame({'Metrics' : ['R2 Score', 'Mean Absolute Error (MAE)', 'Mean Absolute Percentage Error (MAPE)'],
                            'Values' : [f"{r2:.2f}", f"{mae:.2f}",  f"{mape:.2f}"]})

    st.dataframe(results_df, hide_index = True, use_container_width=True)

def st_shap(plot, height=None, width=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height, width=width, scrolling=True)

with col4:
    with st.popover('Features Importance'):
        # Plot feature importance
        # Get feature names
        lgbm_feat_imp = model.feature_importances_ / model.feature_importances_.sum()
        # Create a DataFrame for feature importances
        feat_imp_df = pd.DataFrame({'Feature': model.feature_name_ , 'Importance': lgbm_feat_imp})

        # Sort by importance
        temp_imp = feat_imp_df.sort_values(by='Importance', ascending=False).head(10)
        temp_imp.sort_values(by='Importance', inplace=True)

        # Plot using Plotly
        fig = px.bar(temp_imp, x='Importance', y='Feature', orientation='h', title='Top 10 Feature Importances')
        
        st.plotly_chart(fig)

    with st.popover('Shape Values'):
        # Plot SHAP values
        # Load SHAP values
        temp_shap = temp[model.feature_name_].tail(1)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(temp_shap[model.feature_name_])
        # Create a DataFrame for SHAP values
        # visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
        st_shap(shap.force_plot(shap.Explanation(values=shap_values[0], base_values=explainer.expected_value, data=temp_shap.iloc[0])), height=150, width=1000)
