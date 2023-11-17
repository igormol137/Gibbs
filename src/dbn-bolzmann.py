# Restricted Boltzmann Machine for Time Series Analysis
#
# Igor Mol <igor.mol@makes.ai>
#
# In this implementation, a Restricted Boltzmann Machine (RBM) is employed for 
# time series data generation and reconstruction. The code begins by defining a 
# class, TimeSeriesData, that handles the preprocessing of time series data. The 
# data is loaded from a CSV file and is then normalized using Min-Max scaling. 
# Subsequently, sequences are created from the time series data, with each se-
# quence having a length of 10, suitable for training. This processed data is 
# utilized for training and testing an RBM model. The RBM class, represented by 
# the RBM class, is initialized with the visible and hidden layer sizes, and it 
# includes methods for sampling hidden and visible layer states, as well as 
# training the RBM using contrastive divergence.
#     The RBM model is trained on the preprocessed time series data using the 
# contrastive divergence algorithm. The training involves both positive and ne-
# gative phases, where hidden layer probabilities and states are sampled in the
# positive phase and visible layer probabilities and states are sampled in the 
# negative phase. The model's weights and biases are then updated based on the 
# associations computed during these phases. This process is iterated for a spe-
# cified number of epochs, refining the RBM's ability to learn and represent 
# patterns in the time series data. After training, the RBM is employed to gene-
# rate new samples by initializing visible layer states and sampling subsequent 
# hidden and visible layer states iteratively.
#     Finally, the generated samples are denormalized, transforming them back to 
# the original scale. The create_table function is used to present a tabulated 
# view of the generated and actual time series sequences, facilitating a quali-
# tative assessment of the RBM's performance. The code concludes by executing the
# main function, which orchestrates the entire process, including data prepro-
# cessing, RBM training, generation of new samples, denormalization, and result 
# visualization. The RBM's ability to capture temporal dependencies in time 
# series data is demonstrated through its generative capabilities, providing a 
# valuable tool for various applications, such as synthetic data generation and 
# anomaly detection.


import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tabulate import tabulate

class TimeSeriesData:
    def __init__(self, file_path):
        # Initialize the class with the file path to the time series data
        self.time_series_df = pd.read_csv(file_path)
        self.sequence_length = 10
        self.scaler = MinMaxScaler()
        self.X_train = None
        self.X_test = None

    def preprocess_data(self):
        # Preprocess the time series data by converting the 'Date' column to datetime
        # Scale the 'Value' column using Min-Max scaling
        # Convert the time series data into sequences suitable for training
        self.time_series_df['Date'] = pd.to_datetime(self.time_series_df['Date'])
        values = self.time_series_df['Value'].values.reshape(-1, 1)
        values_scaled = self.scaler.fit_transform(values)

        X = []
        for i in range(len(values_scaled) - self.sequence_length):
            X.append(values_scaled[i:i + self.sequence_length].flatten())

        X = np.array(X)
        train_size = int(len(values_scaled) * 0.8)
        self.X_train, self.X_test = X[:train_size], X[train_size:]

class RBM:
    def __init__(self, visible_size, hidden_size):
        # Initialize the Restricted Boltzmann Machine (RBM) with visible and hidden layer sizes
        self.visible_size = visible_size
        self.hidden_size = hidden_size
        self.weights = np.random.randn(visible_size, hidden_size)
        self.visible_bias = np.zeros((1, visible_size))
        self.hidden_bias = np.zeros((1, hidden_size))

    def sigmoid(self, x):
        # Sigmoid activation function
        return 1 / (1 + np.exp(-x))

    def sample_hidden(self, visible_probs):
        # Sample hidden layer states based on visible layer probabilities
        hidden_probs = self.sigmoid(np.dot(visible_probs, self.weights) + self.hidden_bias)
        hidden_states = np.random.binomial(1, hidden_probs)
        return hidden_probs, hidden_states

    def sample_visible(self, hidden_probs):
        # Sample visible layer states based on hidden layer probabilities
        visible_probs = self.sigmoid(np.dot(hidden_probs, self.weights.T) + self.visible_bias)
        visible_states = np.random.binomial(1, visible_probs)
        return visible_probs, visible_states

    def train(self, data, learning_rate=0.01, epochs=50, batch_size=32):
        # Train the RBM using contrastive divergence
        num_samples = data.shape[0]

        for epoch in range(epochs):
            np.random.shuffle(data)

            for i in range(0, num_samples, batch_size):
                batch_data = data[i:i + batch_size]

                # Positive phase
                positive_hidden_probs, positive_hidden_states = self.sample_hidden(batch_data)
                positive_associations = np.dot(batch_data.T, positive_hidden_probs)

                # Negative phase
                negative_visible_probs, negative_visible_states = self.sample_visible(positive_hidden_states)
                negative_hidden_probs, negative_hidden_states = self.sample_hidden(negative_visible_states)
                negative_associations = np.dot(negative_visible_states.T, negative_hidden_probs)

                # Update weights and biases
                self.weights += learning_rate * (positive_associations - negative_associations) / batch_size
                self.visible_bias += learning_rate * np.mean(batch_data - negative_visible_probs, axis=0)
                self.hidden_bias += learning_rate * np.mean(positive_hidden_probs - negative_hidden_probs, axis=0)

    def generate_samples(self, num_samples):
        # Generate new samples from the RBM
        samples = np.random.rand(num_samples, self.visible_size)
        hidden_probs, _ = self.sample_hidden(samples)
        visible_probs, _ = self.sample_visible(hidden_probs)
        return visible_probs

def create_table(actual, predicted):
    # Create a table comparing actual and predicted values
    table_data = {'Actual': actual.flatten(), 'Predicted': predicted.flatten()}
    table = pd.DataFrame(table_data)
    return tabulate(table, headers='keys', tablefmt='pretty', showindex=False)

def main():
    # Main function
    file_path = "/Users/igormol/Desktop/time_series_data.csv"  # Update with the actual file path
    time_series_data = TimeSeriesData(file_path)
    time_series_data.preprocess_data()

    visible_size = time_series_data.X_train.shape[1]
    hidden_size = 50

    rbm = RBM(visible_size, hidden_size)
    rbm.train(time_series_data.X_train, epochs=50, batch_size=32)

    num_samples = time_series_data.X_test.shape[0]
    generated_samples = rbm.generate_samples(num_samples)

    generated_samples_denormalized = time_series_data.scaler.inverse_transform(generated_samples)
    table = create_table(time_series_data.X_test, generated_samples_denormalized)
    print(table)

if __name__ == "__main__":
    main()
