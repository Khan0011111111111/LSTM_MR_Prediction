import streamlit as st
import numpy as np
import joblib
from keras.models import load_model

# Load the scaler and the LSTM model
scaler = joblib.load('scaler.pkl')
model = load_model('your_model.h5')

# Title of the app
st.title('LSTM Model - Predict MR')
# Subtitle
st.subheader('Developed by: Muhammad Kashif')


# Instructions
st.write("Please input the following parameters to predict MR:")

# Input fields for the parameters
WDC = st.number_input('WDC', min_value=0.0, value=1.0)
CSAFR = st.number_input('CSAFR', min_value=0.0, value=1.0)
DMR = st.number_input('DMR', min_value=0.0, value=1.0)
σ3 = st.number_input('σ3', min_value=0.0, value=1.0)
σd = st.number_input('σd', min_value=0.0, value=1.0)

# Collect the input values
input_data = np.array([[WDC, CSAFR, DMR, σ3, σd]])

# Scale the input data using the saved scaler
input_scaled = scaler.transform(input_data)

# Reshape the input for the LSTM model
input_scaled = input_scaled.reshape((input_scaled.shape[0], 1, input_scaled.shape[1]))

# Custom CSS for button styling
st.markdown("""
    <style>
    div.stButton > button {
        background-color: orange;
        color: white;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
        font-weight: bold;
        border-radius: 5px;
        transition: background-color 0.3s ease;
    }
    div.stButton > button:active {
        background-color: black;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# Predict MR when the user presses the button
if st.button('Predict MR'):
    prediction = model.predict(input_scaled)
    st.markdown(f'<h3 style="color:orange;">Predicted MR: {prediction[0][0]:.2f}</h3>', unsafe_allow_html=True)
