# Next-Frame Approach to Model Time Series
#
# Igor Mol <igor.mol@makes.ai>
#
# This program implements a class called TimeSeriesModel to perform time series 
# modeling according to the so-called Next-Frame Technique. We employ Long Short
# Term Memory (LSTM) neural networks. The program reads time series data from a 
# CSV file, normalizes it using Min-Max scaling, and then converts it into 
# sequences of input-output pairs suitable for training an LSTM model. The LSTM 
# model is built and trained using the Keras library, aiming to predict future 
# values in the time series. After training, the model generates predictions on 
# a test set. These predictions and the actual values are denormalized to their 
# original scale. The program prints a table comparing the predicted and 
# actual values and plots the time series to visually assess the model's perfor-
# mance. This approach aids in understanding and forecasting patterns in time 
# series data, providing insights into potential future trends. The use of LSTM 
# networks allows the model to capture long-term dependencies and patterns in the
# time series, making it effective for various prediction tasks.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tabulate import tabulate

# Define a class 'TimeSeriesModel' to encapsulate the functionality of time series modeling
class TimeSeriesModel:
    def __init__(self, file_path, sequence_length=10):
        self.file_path = file_path
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler()

    # Load time series data from a CSV file and return the 'Value' column as a numpy array
    def load_data(self):
        time_series_df = pd.read_csv(self.file_path)
        time_series_df['Date'] = pd.to_datetime(time_series_df['Date'])
        values = time_series_df['Value'].values.reshape(-1, 1)
        return values

    # Normalize the input data using Min-Max scaling
    def normalize_data(self, data):
        return self.scaler.fit_transform(data)

    # Create input-output sequences for the LSTM model
    def create_sequences(self, data):
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:i + self.sequence_length])
            y.append(data[i + self.sequence_length])
        return np.array(X), np.array(y)

    # Build and compile an LSTM model with specified architecture
    def build_lstm_model(self):
        model = Sequential()
        model.add(LSTM(50, activation='relu', input_shape=(self.sequence_length, 1)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    # Train the LSTM model with training data
    def train_model(self, model, X_train, y_train, epochs=50, batch_size=32):
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2)
        return model

    # Denormalize the normalized data to obtain the original scale
    def denormalize_data(self, normalized_data):
        return self.scaler.inverse_transform(normalized_data)

    # Generate predictions using the trained LSTM model on test data
    def generate_predictions(self, model, X_test):
        return model.predict(X_test)

    # Print a table comparing predicted and actual values
    def print_results_table(self, y_test_denormalized, y_pred_denormalized):
        results_table = pd.DataFrame({
            'Actual': y_test_denormalized.flatten(),
            'Predicted': y_pred_denormalized.flatten()
        })

        table = tabulate(results_table, headers='keys', tablefmt='fancy_grid', showindex=False)
        print("\nTable with Predicted vs Actual Values:")
        print(table)

    # Plot the actual and predicted time series values
    def plot_time_series(self, y_test_denormalized, y_pred_denormalized):
        plt.figure(figsize=(10, 6))
        plt.plot(y_test_denormalized, label='Actual')
        plt.plot(y_pred_denormalized, label='Predicted')
        plt.legend()
        plt.title('Actual vs. Predicted Time Series')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.show()

# Define the main function to execute the time series modeling
def main():
    file_path = "/Users/igormol/Desktop/time_series_data.csv"  # Update with the actual file path
    time_series_model = TimeSeriesModel(file_path)
    
    # Load and preprocess the time series data
    values = time_series_model.load_data()
    values_scaled = time_series_model.normalize_data(values)

    # Convert the time series data into input-output sequences
    X, y = time_series_model.create_sequences(values_scaled)

    # Split the data into training and testing sets
    train_size = int(len(values_scaled) * 0.8)
    X_train, X_test, y_train, y_test = X[:train_size], X[train_size:], y[:train_size], y[train_size:]

    # Build and train the LSTM model
    lstm_model = time_series_model.build_lstm_model()
    trained_lstm_model = time_series_model.train_model(lstm_model, X_train, y_train)

    # Generate predictions on the test set
    y_pred = time_series_model.generate_predictions(trained_lstm_model, X_test)

    # Denormalize the predictions and actual values
    y_pred_denormalized = time_series_model.denormalize_data(y_pred)
    y_test_denormalized = time_series_model.denormalize_data(y_test)

    # Print results table and plot time series
    time_series_model.print_results_table(y_test_denormalized, y_pred_denormalized)
    time_series_model.plot_time_series(y_test_denormalized, y_pred_denormalized)

# Execute the main function if the script is run
if __name__ == "__main__":
    main()
