import numpy as np
import streamlit as st
from sklearn.metrics import mean_squared_error, root_mean_squared_log_error
import matplotlib.pyplot as plt

from models.reservoir_computer import ReservoirComputer
from src.reservoir_computing.load_aq_data import load_aq_data

# Function to plot results
def plot_results(true_values, predictions):
    plt.figure(figsize=(10, 5))
    plt.plot(true_values, label='True Values')
    plt.plot(predictions, label='Predictions')
    plt.legend()
    st.pyplot(plt)

def trim_array(array):
    # Trim array to be greater than or equal to 0
    negative_mask = array < 0
    negative_count = np.sum(negative_mask)
    array = np.maximum(0, array)  # Element-wise max with 0
    return array

# Streamlit state for data readiness
if 'data_ready' not in st.session_state:
    st.session_state.data_ready = False

# Streamlit interface
st.title("Reservoir Computer Visualization")

# Data options
st.sidebar.title("Parameters")

st.sidebar.markdown("---")

st.sidebar.header("Data Options")
st.sidebar.write("This imports and processes the Air Quality dataset from the UCI Machine Learning Repository. (link: https://archive.ics.uci.edu/ml/datasets/Air+Quality)")
data_delay = st.sidebar.slider("Data Delay", min_value=1, max_value=10, value=5)

if st.sidebar.button("Process Data"):
    with st.spinner("Processing Data..."):
        input_dim, (train_input, test_input), (train_output, test_output), target_scaler = load_aq_data('data/air_quality/AirQualityUCI.csv', delay=5)
        st.session_state.data_ready = True
        st.session_state.input_dim = input_dim
        st.session_state.train_input = train_input
        st.session_state.test_input = test_input
        st.session_state.train_output = train_output
        st.session_state.test_output = test_output
        st.session_state.target_scaler = target_scaler
        st.success("Data Processed Successfully!")

st.sidebar.markdown("---")

# Input parameters
st.sidebar.header("Reservoir Computer Parameters")
reservoir_size = st.sidebar.number_input("Reservoir Size", min_value=10, value=100)
spectral_radius = st.sidebar.slider("Spectral Radius", min_value=0.1, max_value=1.0, value=0.95)
sparsity = st.sidebar.slider("Sparsity", min_value=0.0, max_value=1.0, value=0.05)
ridge_alpha = st.sidebar.number_input("Ridge Alpha", min_value=1e-10, value=1e-6, format="%.7f", step=1e-7)

# Train and predict
if st.sidebar.button("Train and Predict"):

    if not st.session_state.data_ready:
        st.error("Please process the data first!")
        st.stop()

    with st.spinner("Training and Predicting..."):

        input_dim = st.session_state.input_dim
        train_input = st.session_state.train_input
        test_input = st.session_state.test_input
        train_output = st.session_state.train_output
        test_output = st.session_state.test_output
        target_scaler = st.session_state.target_scaler
        
        reservoir_computer = ReservoirComputer(
            input_dim=input_dim,
            reservoir_size=reservoir_size,
            output_dim=1,
            spectral_radius=spectral_radius,
            sparsity=sparsity,
            ridge_alpha=ridge_alpha
        )
        
        reservoir_computer.train(train_input, train_output)
        predictions = reservoir_computer.predict(test_input)
        
        def inverse_transform_predictions(scaled_predictions):
            return target_scaler.inverse_transform(scaled_predictions.reshape(-1, 1)).flatten()

        predictions = inverse_transform_predictions(predictions)
        predictions = trim_array(predictions)

        test_output = inverse_transform_predictions(test_output)
        test_output = trim_array(test_output)

        # Calculate Mean Squared Error (MSE)
        mse = mean_squared_error(test_output, predictions)
        rmsle = root_mean_squared_log_error(test_output, predictions)

        plot_results(test_output, predictions)

        st.write(f"Mean Squared Error: {mse}")
        st.write(f"RMSLE: {rmsle}")