# Import necessary libraries
#
# Igor Mol <igor.mol@makes.ai>
#
# The provided Python code implements a Restricted Boltzmann Machine (RBM) for 
# time series data generation. It begins by defining a TimeSeriesData class res-
# ponsible for preprocessing the input time series data. This involves converting 
# the 'Date' column to datetime, scaling the 'Value' column using MinMaxScaler, 
# and creating sequences of data to train the RBM. The RBM is implemented in the 
# RBM class, which has methods for training the model using contrastive diver-
# gence and generating samples. The training process involves iterating through 
# epochs, shuffling the data, and updating weights and biases based on the dif-
# ference between positive and negative associations. The generated samples are 
# then denormalized using the scaler. The main function orchestrates the entire 
# process, loading the time series data, preprocessing it, training the RBM, 
# generating samples, denormalizing them, and finally, creating and printing a 
# table that compares the actual and generated time series sequences.
#     The primary goal of this code is to showcase the use of RBMs for time se-
# ries data generation. RBMs are a type of unsupervised learning model, and in 
# this context, they are used to learn the underlying patterns and structures in
# the time series data. The generated samples are denormalized and presented in 
# tabulated form, allowing for a direct visual comparison between the actual and
# generated sequences. The training process involves iteratively adjusting the 
# RBM's parameters to capture the statistical dependencies within the input data
# ultimately enabling the model to generate synthetic time series sequences that 
# exhibit similar patterns to the original data. This approach is particularly 
# useful for tasks such as anomaly detection, data augmentation, or creating 
# realistic synthetic data for testing machine learning models.
#     In more detail, the TimeSeriesData class handles the initial loading and 
# preprocessing of time series data. It reads the data from a CSV file, trans-
# forms the 'Date' column to datetime, scales the 'Value' column using MinMax-
# Scaler, and creates sequences of data suitable for training the RBM. The RBM 
# is implemented in the RBM class, where the sigmoid activation function is de-
# fined along with methods for sampling hidden and visible layers. The training
# process involves both positive and negative phases, where hidden layer states 
# are sampled based on visible layer states, and vice versa. The weights and 
# biases are updated using contrastive divergence, a technique specific to RBMs.
#     Finally, the main function orchestrates the entire workflow, from loading 
# the data to training the RBM, generating denormalized samples, and creating a 
# table that compares the actual and generated time series sequences using the 
# 'tabulate' library for a more readable output.

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tabulate import tabulate

# Define the TimeSeriesData class for data preprocessing
class TimeSeriesData:
    def __init__(self, file_path):
        # Initialize with file path, sequence length, MinMaxScaler, and placeholders for training and testing data
        self.time_series_df = pd.read_csv(file_path)
        self.sequence_length = 10
        self.scaler = MinMaxScaler()
        self.X_train = None
        self.X_test = None

    # Preprocess the time series data
    def preprocess_data(self):
        # Convert 'Date' column to datetime, extract 'Value' column, and scale using MinMaxScaler
        self.time_series_df['Date'] = pd.to_datetime(self.time_series_df['Date'])
        values = self.time_series_df['Value'].values.reshape(-1, 1)
        values_scaled = self.scaler.fit_transform(values)

        # Create sequences for training the RBM
        X = []
        for i in range(len(values_scaled) - self.sequence_length):
            X.append(values_scaled[i:i + self.sequence_length].flatten())
        X = np.array(X)

        # Split the data into training and testing sets
        train_size = int(len(values_scaled) * 0.8)
        self.X_train, self.X_test = X[:train_size], X[train_size:]

# Define the RBM class for Restricted Boltzmann Machine implementation
class RBM:
    def __init__(self, visible_size, hidden_size):
        # Initialize RBM parameters: weights, visible bias, and hidden bias
        self.visible_size = visible_size
        self.hidden_size = hidden_size
        self.weights = np.random.randn(visible_size, hidden_size)
        self.visible_bias = np.zeros((1, visible_size))
        self.hidden_bias = np.zeros((1, hidden_size))

    # Sigmoid activation function
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Sample hidden layer probabilities and states
    def sample_hidden(self, visible_probs):
        hidden_probs = self.sigmoid(np.dot(visible_probs, self.weights) + self.hidden_bias)
        hidden_states = np.random.binomial(1, hidden_probs)
        return hidden_probs, hidden_states

    # Sample visible layer probabilities and states
    def sample_visible(self, hidden_probs):
        visible_probs = self.sigmoid(np.dot(hidden_probs, self.weights.T) + self.visible_bias)
        visible_states = np.random.binomial(1, visible_probs)
        return visible_probs, visible_states

    # Train the RBM using contrastive divergence
    def train(self, data, learning_rate=0.01, epochs=50, batch_size=32):
        num_samples = data.shape[0]

        for epoch in range(epochs):
            np.random.shuffle(data)

            for i in range(0, num_samples, batch_size):
                batch_data = data[i:i + batch_size]

                # Positive phase: Sample hidden layer states based on visible layer states
                positive_hidden_probs, positive_hidden_states = self.sample_hidden(batch_data)
                positive_associations = np.dot(batch_data.T, positive_hidden_probs)

                # Negative phase: Sample visible and hidden layer states iteratively
                negative_visible_probs, negative_visible_states = self.sample_visible(positive_hidden_states)
                negative_hidden_probs, negative_hidden_states = self.sample_hidden(negative_visible_states)
                negative_associations = np.dot(negative_visible_states.T, negative_hidden_probs)

                # Update weights and biases based on contrastive divergence
                self.weights += learning_rate * (positive_associations - negative_associations) / batch_size
                self.visible_bias += learning_rate * np.mean(batch_data - negative_visible_probs, axis=0)
                self.hidden_bias += learning_rate * np.mean(positive_hidden_probs - negative_hidden_probs, axis=0)

    # Generate samples using the trained RBM
    def generate_samples(self, num_samples):
        samples = np.random.rand(num_samples, self.visible_size)
        hidden_probs, _ = self.sample_hidden(samples)
        visible_probs, _ = self.sample_visible(hidden_probs)
        return visible_probs

# Function to create a tabulated view of actual and predicted time series sequences
def create_table(actual, predicted):
    table_data = {'Actual': actual.flatten(), 'Predicted': predicted.flatten()}
    table = pd.DataFrame(table_data)
    return tabulate(table, headers='keys', tablefmt='pretty', showindex=False)

# Main function
def main():
    # File path for time series data
    file_path = "/Users/igormol/Desktop/time_series_data.csv"
    
    # Create an instance of TimeSeriesData and preprocess the data
    time_series_data = TimeSeriesData(file_path)
    time_series_data.preprocess_data()

    # Set visible and hidden layer sizes for RBM
    visible_size = time_series_data.X_train.shape[1]
    hidden_size = 50

    # Create an instance of RBM and train it on the preprocessed data
    rbm = RBM(visible_size, hidden_size)
    rbm.train(time_series_data.X_train, epochs=50, batch_size=32)

    # Generate samples using the trained RBM
    num_samples = time_series_data.X_test.shape[0]
    generated_samples = rbm.generate_samples(num_samples)

    # Denormalize generated samples
    generated_samples_denormalized = time_series_data.scaler.inverse_transform(generated_samples)

    # Create and print a table comparing actual and generated time series sequences
    table = create_table(time_series_data.X_test, generated_samples_denormalized)
    print(table)

# Execute main function if the script is run
if __name__ == "__main__":
    main()
