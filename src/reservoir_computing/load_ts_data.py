import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from category_encoders import OrdinalEncoder

def simple_embeddings(series, dim):
    unique_values = series.unique()
    embeddings_dict = {val: np.random.randn(dim) for val in unique_values}
    return pd.DataFrame(series.map(embeddings_dict).tolist(), index=series.index)

def process_data(main_data_dir: str, data_dir: str):
    # Load data
    main_df = pd.read_csv(main_data_dir)
    
    stores = pd.read_csv(data_dir + 'stores.csv')
    oil = pd.read_csv(data_dir + 'oil.csv')
    holidays = pd.read_csv(data_dir + 'holidays_events.csv')

    main_df['date'] = pd.to_datetime(main_df['date'])

    # Merge with stores data
    main_df = pd.merge(main_df, stores, on='store_nbr', how='left')

    # Process oil data
    oil['date'] = pd.to_datetime(oil['date'])
    oil['dcoilwtico'] = oil['dcoilwtico'].interpolate()
    oil = oil.rename(columns={'dcoilwtico': 'oil_price'})
    main_df = pd.merge(main_df, oil[['date', 'oil_price']], on='date', how='left')
    main_df['oil_price'] = main_df['oil_price'].ffill().bfill()

    # Process holiday data
    holidays['is_holiday'] = 1
    holidays['date'] = pd.to_datetime(holidays['date'])
    main_df = pd.merge(main_df, holidays[['date', 'is_holiday']], on='date', how='left')
    main_df['is_holiday'] = main_df['is_holiday'].fillna(0)

    # Create time-based features
    main_df['day_of_week'] = main_df['date'].dt.day_of_week
    main_df['month'] = main_df['date'].dt.month
    main_df['year'] = main_df['date'].dt.year

    # Apply ordinal encoding to 'type'
    ordinal_encoder = OrdinalEncoder(cols=['type'], mapping=[{'col': 'type', 'mapping': {'A':1, 'B':2, 'C':3, 'D':4, 'E':5}}])
    main_df['type_encoded'] = ordinal_encoder.fit_transform(main_df[['type']])

    # Create embeddings for high cardinality categorical features
    family_embedded = simple_embeddings(main_df['family'], dim=8)
    main_df = pd.concat([family_embedded.add_prefix('family_'), main_df], axis=1)
    
    main_df = main_df.sort_values("date")
    main_df = main_df.drop_duplicates()
    main_df = main_df.drop(columns=["id", "family", "city", "state", "type", "date"])
    
    # Separate target variable
    target = main_df['sales']
    features = main_df.drop(columns=['sales'])

    # Scale features (not including 'id' and 'date')
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Scale the target variable separately
    target_scaler = StandardScaler()
    target_scaled = target_scaler.fit_transform(target.values.reshape(-1, 1))

    return features_scaled, target_scaled, target_scaler

def load_ts_data(data_dir: str, num_rows: int, delay: int = 5):
    # Load data
    train_dir = data_dir + 'train.csv'
    test_dir = data_dir + 'test.csv'

    train_features, train_target, target_scaler = process_data(train_dir, data_dir)

    if num_rows is not None:
        train_features = train_features[:num_rows]
        train_target = train_target[:num_rows]

    # Convert to numpy arrays
    train_features = np.array(train_features)
    train_target = np.array(train_target)

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