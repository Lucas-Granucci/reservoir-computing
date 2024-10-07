import numpy as np
from src.reservoir_computing.load_aq_data import load_aq_data
from src.reservoir_computing.load_ts_data import load_ts_data

from train import train_reservoir_computer

# Load data
#data_dir = 'data/air_quality/AirQualityUCI.csv'
#input_dim, (train_input, test_input), (train_output, test_output), target_scaler = load_aq_data(data_dir, delay=5)

data_dir = 'data/store-sales/'
input_dim, (train_input, test_input), (train_output, test_output), target_scaler = load_ts_data(data_dir, num_rows=5000, delay=10)

rc_params = {
    'input_dim': input_dim,
    'reservoir_size': 500,
    'reservoir_ridge_alpha': 1e-6,
    'reservoir_output_dim': 1,
}

trained_rc, mse, rmsle = train_reservoir_computer(rc_params, train_input, test_input, train_output, test_output, target_scaler, save_path="output/sales_predictions.png")

print(f"Mean Squared Error: {mse}")
print(f"RMSLE: {rmsle}")