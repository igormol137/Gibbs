# Hopefield Network for Time-series Completion
#
# Igor Mol <igor.mol@makes.ai>
#
# Abstract:
# The approach implemented in the following program demonstrates the application 
# of the Hopfield Networks for time-series completion, offering a systematic 
# framework for modeling and predicting missing values in sequential data. 
#     A class named HopfieldNetwork is utilized to model and predict missing 
# values in a time series. The network is initialized with a specified size, and
# the weights matrix, representing the connections between neurons, is set to 
# zeros initially. The training method updates these weights based on the outer 
# product of input patterns, excluding self-connections to prevent distortion.
#     The prediction process involves iteratively updating the output pattern 
# using the dot product with the weights matrix. This updated pattern is deter-
# mined by applying a sign function to the dot product result. The number of 
# iterations is user-defined, influencing the convergence of the predicted 
# pattern.
#
# "The memory has already entered your consciousness, but you must find it. It 
# will appear in dreams, in your waking hours, when you turn the page of a book 
# or a corner. Do not be impatient, do not invent memories. Chance might favor 
# or delay you, in its own mysterious way. As I begin to forget, you will begin 
# to remember. I promise nothing more".
# - Jorge Luis Borges

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Class `init':
# - Initializes the Hopfield Network with a specified size.
# - Sets the size of the network and initializes the weights matrix with zeros.

class HopfieldNetwork:
    
    def __init__(self, size):
        self.size = size
        self.weights = np.zeros((size, size))

# train:
# - Trains the Hopfield Network with a set of input patterns.
# - Updates the weights matrix based on the outer product of each pattern with itself.
# - Sets diagonal elements of the weights matrix to zero to prevent self-connections.
    
    def train(self, patterns):
        for pattern in patterns:
            self.weights += np.outer(pattern, pattern)
            np.fill_diagonal(self.weights, 0)

# predict:
# - Predicts a pattern using the trained Hopfield Network.
# - Iteratively updates the output pattern based on the dot product with the weights matrix.
# - Applies a sign function to the dot product result.
# - Returns the predicted pattern after a specified number of iterations.
    
    def predict(self, input_pattern, max_iterations=100):
        output_pattern = np.copy(input_pattern)
        for _ in range(max_iterations):
            output_pattern = np.sign(np.dot(self.weights, output_pattern))
        return output_pattern
        
# Two utility functions, normalize_data and denormalize_pattern, assist in pre-
# processing the time series. normalize_data scales the data to the [0, 1] ran-
# ge, while denormalize_pattern reverts a normalized pattern to its original 
# scale based on the original minimum and maximum values.

def normalize_data(data):
    min_value, max_value = np.min(data), np.max(data)
    normalized_data = (data - min_value) / (max_value - min_value)
    return normalized_data, min_value, max_value

def denormalize_pattern(pattern, min_value, max_value):
    return pattern * (max_value - min_value) + min_value

# In the main function, time series data is loaded from a CSV file and subse-
# quently normalized using the utility functions. A Hopfield Network instance is 
# created, and the network is trained with the normalized time series values. A 
# subset of the time series, such as the first half, is chosen as the input 
# pattern.
#     The Hopfield Network is then employed to predict the remaining values in 
# the time series. The predicted and input patterns are denormalized to their 
# original scale, and the results are presented in a tabular format. Additiona-
# lly, a graphical representation is provided through a time series plot that 
# showcases the actual, input, and predicted values.

def main():
    # Load the time-series data
    file_path = "/Users/igormol/Desktop/time_series_data.csv"
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date')

    # Normalize the 'Value' column
    values = df['Value'].values
    normalized_values, min_value, max_value = normalize_data(values)

    # Create a Hopfield Network
    input_size = len(normalized_values)
    hopfield_net = HopfieldNetwork(size=input_size)

    # Train the network with the normalized values
    hopfield_net.train([normalized_values])

    # Choose a subset of the time series as input (e.g., the first half)
    input_pattern = normalized_values[:input_size]

    # Predict the remaining values
    predicted_pattern = hopfield_net.predict(input_pattern)

    # Denormalize the patterns
    input_pattern_denormalized = denormalize_pattern(input_pattern, min_value, max_value)
    predicted_pattern_denormalized = denormalize_pattern(predicted_pattern, min_value, max_value)

    # Create a DataFrame for the results
    results = pd.DataFrame({'Date': df['Date'], 'Actual': values, 'Input': input_pattern_denormalized, 'Predicted': predicted_pattern_denormalized})

    # Print the results in a formatted table
    print(results)

    # Plot the actual, input, and predicted time series
    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], values, label='Actual', marker='o')
    plt.plot(df['Date'], input_pattern_denormalized, label='Input', marker='o')
    plt.plot(df['Date'], predicted_pattern_denormalized, label='Predicted', marker='o')
    plt.title('Hopfield Network Time Series Completion')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
