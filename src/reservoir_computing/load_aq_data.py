import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def process_data(data_dir: str):
    # Load data
    data = pd.read_csv(data_dir, sep=';', decimal=',', na_values=-200)
    
    # Combine 'Date' and 'Time' columns into a single datetime column
    data['Date_Time'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], format='%d/%m/%Y %H.%M.%S')
    
    # Drop the original 'Date' and 'Time' columns
    data = data.drop(['Date', 'Time'], axis=1)
    
    # Drop columns with too many missing values and the last two columns which are not features
    data = data.drop(['Date_Time', 'NMHC(GT)', 'Unnamed: 15', 'Unnamed: 16'], axis=1)
    
    # Drop rows with missing values
    data = data.dropna()
    
    # Separate the target variable (CO)
    target = data['CO(GT)']
    features = data.drop('CO(GT)', axis=1)
    
    # Normalize the features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Scale the target variable separately
    target_scaler = StandardScaler()
    target_scaled = target_scaler.fit_transform(target.values.reshape(-1, 1))

    return features_scaled, target_scaled, target_scaler


def load_aq_data(data_dir: str, delay: int = 5):

    train_features, train_target, target_scaler = process_data(data_dir)
    
    # Create time-delayed input and output data
    def create_time_delay_data(features, target, delay):
        delayed_input = []
        delayed_output = []
        for i in range(delay, len(features)):
            delayed_input.append(features[i-delay:i].flatten())
            delayed_output.append(target[i])
        return np.array(delayed_input), np.array(delayed_output)
    
    input_data, output_data = create_time_delay_data(train_features, train_target, delay)
    
    # Set up input dimension
    input_dim = delay * train_features.shape[1]
    
    # Split the data into training and testing sets
    split_index = int(0.8 * len(input_data))
    train_input, test_input = input_data[:split_index], input_data[split_index:]
    train_output, test_output = output_data[:split_index], output_data[split_index:]
    
    input_batch = (train_input, test_input)
    output_batch = (train_output, test_output)
    
    return input_dim, input_batch, output_batch, target_scaler