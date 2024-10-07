import numpy as np
from models.reservoir_computer import ReservoirComputer

class MultiReservoirComputer:
    """
    A class to represent a multi-reservoir computer, which is an ensemble of multiple reservoir computers.
    Attributes:
    -----------
    input_dim : int
        The dimension of the input data.
    reservoir_sizes : list of int
        A list containing the sizes of each reservoir in the ensemble.
    output_dim : int, optional
        The dimension of the output data (default is 1).
    ridge_alpha : float, optional
        The regularization parameter for ridge regression (default is 1e-6).
    reservoirs : list of ReservoirComputer
        A list of ReservoirComputer instances, one for each size specified in reservoir_sizes.
    Methods:
    --------
    train(train_input, train_output):
        Trains each reservoir in the ensemble using the provided training data.
    predict(test_input):
        Predicts the output for the given test input by averaging the predictions from each reservoir in the ensemble.
    """
    def __init__(self, input_dim, reservoir_sizes, output_dim=1, ridge_alpha=1e-6):
        self.input_dim = input_dim
        self.reservoir_sizes = reservoir_sizes
        self.output_dim = output_dim
        self.ridge_alpha = ridge_alpha
        self.reservoirs = [ReservoirComputer(input_dim, size, output_dim, ridge_alpha) for size in reservoir_sizes]
    
    def train(self, train_input, train_output):
        for reservoir in self.reservoirs:
            reservoir.train(train_input, train_output)
    
    def predict(self, test_input):
        predictions = [reservoir.predict(test_input) for reservoir in self.reservoirs]
        predictions = np.mean(predictions, axis=0)
        return predictions