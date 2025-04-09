import streamlit as st
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb

# Load trained model
with open("models/xgboost_optuna_model.pkl", "rb") as f:
    model = pickle.load(f)

# Define the final feature list
FEATURES = [
    'feature_0', 'feature_1', 'feature_10', 'feature_11', 'feature_12', 'feature_13', 'feature_14',
    'feature_15', 'feature_16', 'feature_17', 'feature_18', 'feature_19', 'feature_2', 'feature_3',
    'feature_4', 'feature_5', 'feature_6', 'feature_7', 'feature_8', 'feature_9',
    'feature_0_x_feature_1', 'feature_0_div_feature_10', 'feature_0_div_feature_15',
    'feature_11_x_feature_13', 'feature_11_sub_feature_18', 'feature_11_sub_feature_19',
    'feature_11_x_feature_2', 'feature_13_x_feature_2', 'feature_13_x_feature_9',
    'feature_13_div_feature_9', 'feature_13_sub_feature_9',
    'feature_18_div_feature_2', 'feature_18_sub_feature_2'
]

# Define min/max values for sliders
feature_ranges = {
    'feature_0': (0, 1000),
    'feature_1': (0, 600),
    'feature_10': (0, 800),
    'feature_11': (0, 10),
    'feature_12': (0, 1000),
    'feature_13': (0, 210),
    'feature_14': (0, 60),
    'feature_15': (0, 600),
    'feature_16': (0, 200),
    'feature_17': (0, 140),
    'feature_18': (0, 60),
    'feature_19': (0, 1000),
    'feature_2': (0, 650),
    'feature_3': (0, 550),
    'feature_4': (0, 1000),
    'feature_5': (0, 650),
    'feature_6': (0, 450),
    'feature_7': (0, 810),
    'feature_8': (0, 400),
    'feature_9': (0, 400)
}
# Streamlit app
st.set_page_config(layout="wide")
st.title("Wizeline Target Prediction App")

# Layout: sliders on the left, prediction center
col1, col2, _ = st.columns([1.5, 1, 0.5])

# Collect input features in col1
with col1:
    st.markdown("### Input Features")
    input_data = {}
    for feature in FEATURES:
        if 'x_' in feature or 'div_' in feature or 'sub_' in feature:
            continue
        f_min, f_max = feature_ranges.get(feature, (-100.0, 100.0))
        input_data[feature] = st.slider(f"{feature}", min_value=float(f_min), max_value=float(f_max), value=float((f_min + f_max) / 2))

# Generate interaction features
data = input_data.copy()
data['feature_0_x_feature_1'] = data['feature_0'] * data['feature_1']
data['feature_0_div_feature_10'] = data['feature_0'] / (data['feature_10'] + 1e-6)
data['feature_0_div_feature_15'] = data['feature_0'] / (data['feature_15'] + 1e-6)
data['feature_11_x_feature_13'] = data['feature_11'] * data['feature_13']
data['feature_11_sub_feature_18'] = data['feature_11'] - data['feature_18']
data['feature_11_sub_feature_19'] = data['feature_11'] - data['feature_19']
data['feature_11_x_feature_2'] = data['feature_11'] * data['feature_2']
data['feature_13_x_feature_2'] = data['feature_13'] * data['feature_2']
data['feature_13_x_feature_9'] = data['feature_13'] * data['feature_9']
data['feature_13_div_feature_9'] = data['feature_13'] / (data['feature_9'] + 1e-6)
data['feature_13_sub_feature_9'] = data['feature_13'] - data['feature_9']
data['feature_18_div_feature_2'] = data['feature_18'] / (data['feature_2'] + 1e-6)
data['feature_18_sub_feature_2'] = data['feature_18'] - data['feature_2']

# Make prediction in col2
m = st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: rgb(204, 49, 49);
}
</style>""", unsafe_allow_html=True)
with col2:
    st.markdown("# Prediction \n Click on the predict button to get predictions for the target.")
    if st.button("Predict"):
        X = pd.DataFrame([data])[FEATURES]
        dmatrix = xgb.DMatrix(X)
        prediction = model.predict(dmatrix)[0]
        st.success(f"Predicted target: {prediction:.2f}")