import numpy as np
import matplotlib.pyplot as plt
from models.reservoir_computer import ReservoirComputer
from sklearn.metrics import mean_squared_error, root_mean_squared_log_error

def trim_array(array):
    # Trim array to be greater than or equal to 0
    negative_mask = array < 0
    negative_count = np.sum(negative_mask)
    array = np.maximum(0, array)  # Element-wise max with 0
    return array

def train_reservoir_computer(rc_params, train_input, test_input, train_output, test_output, target_scaler, save_path=None):

    def inverse_transform_predictions(scaled_predictions):
        return target_scaler.inverse_transform(scaled_predictions.reshape(-1, 1)).flatten()
    
    # Initialize the reservoir computer
    reservoir_input_dim = rc_params['input_dim']
    reservoir_size = rc_params['reservoir_size']
    reservoir_ridge_alpha = rc_params['reservoir_ridge_alpha']
    reservoir_output_dim = rc_params['reservoir_output_dim']

    reservoir_computer = ReservoirComputer(reservoir_input_dim, reservoir_size, output_dim=reservoir_output_dim, ridge_alpha=reservoir_ridge_alpha)

    # Train the reservoir computer
    reservoir_computer.train(train_input, train_output)

    # Make predictions
    predictions = reservoir_computer.predict(test_input)

    predictions = inverse_transform_predictions(predictions)
    predictions = trim_array(predictions)

    test_output = inverse_transform_predictions(test_output)
    test_output = trim_array(test_output)

    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(test_output, predictions)
    rmsle = root_mean_squared_log_error(test_output, predictions)

    if save_path is not None:
        # Plot the predictions
        plt.figure(figsize=(12, 6))
        plt.plot(test_output, label="Actual")
        plt.plot(predictions, label="Predicted")
        plt.title(f"Reservoir Computer Predictions (MSE: {mse:.4f}) (RMSLE: {rmsle:.4f})")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend()
        plt.savefig(save_path)
        plt.close()

    return reservoir_computer, mse, rmsle