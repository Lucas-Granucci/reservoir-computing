import numpy as np
from tqdm import tqdm
from sklearn.linear_model import Ridge

class ReservoirComputer:
    """
    A class to represent a Reservoir Computer for time series prediction.
    Attributes
    ----------
    input_dim : int
        Dimension of the input data.
    reservoir_size : int
        Number of neurons in the reservoir.
    output_dim : int
        Dimension of the output data.
    spectral_radius : float, optional
        Spectral radius of the reservoir weight matrix (default is 0.95).
    sparsity : float, optional
        Sparsity of the reservoir weight matrix (default is 0.05).
    ridge_alpha : float, optional
        Regularization strength for ridge regression (default is 1e-6).
    W_reservoir : np.ndarray
        Weight matrix for the reservoir.
    W_input : np.ndarray
        Weight matrix for the input.
    W_output : np.ndarray
        Weight matrix for the output.
    state : np.ndarray
        Current state of the reservoir.
    Methods
    -------
    _initialize_reservoir():
        Initializes the reservoir weight matrix with given sparsity and spectral radius.
    _update_state(input_vector):
        Updates the state of the reservoir given an input vector.
    train(input_data, output_data):
        Trains the reservoir computer using input and output data.
    predict(input_data):
        Predicts the output for given input data.
    """
    def __init__(self, input_dim, reservoir_size, output_dim, spectral_radius=0.95, sparsity=0.05, ridge_alpha=1e-6):
        self.input_dim = input_dim
        self.reservoir_size = reservoir_size
        self.output_dim = output_dim
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.ridge_alpha = ridge_alpha
        
        # Initialize reservoir weights
        self.W_reservoir = self._initialize_reservoir()

        # Initialize input weights
        self.W_input = np.random.rand(self.reservoir_size, self.input_dim) - 0.5

        # Initialize output weights
        self.W_output = np.zeros((self.output_dim, self.reservoir_size))

        # Reservoir state
        self.state = np.zeros((self.reservoir_size, 1))


    def _initialize_reservoir(self):
        W = np.random.rand(self.reservoir_size, self.reservoir_size) - 0.5
        W[np.random.rand(*W.shape) > self.sparsity] = 0
        spectral_radius = np.max(np.abs(np.linalg.eigvals(W)))
        return W * (self.spectral_radius / spectral_radius)
    
    def _update_state(self, input_vector):
        input_vector = input_vector.reshape(-1, 1)
        preactivation = np.dot(self.W_input, input_vector) + np.dot(self.W_reservoir, self.state)
        self.state = np.tanh(preactivation)

    def train(self, input_data, output_data):
        states = []
        for input_vector in tqdm(input_data, desc="Training"):
            self._update_state(input_vector)
            states.append(self.state.flatten())
        states = np.array(states)

        # Train output weights using ridge regression
        ridge = Ridge(alpha=self.ridge_alpha)
        ridge.fit(states, output_data)
        self.W_output = ridge.coef_

    def predict(self, input_data):
        predictions = []
        for input_vector in tqdm(input_data, desc="Predicting"):
            self._update_state(input_vector)
            prediction = np.dot(self.W_output, self.state).flatten()
            predictions.append(prediction)
        return np.array(predictions)