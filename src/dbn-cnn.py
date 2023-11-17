# Deep Belief Network of Type CNN/LSTM for Time Series Analysis
#
# Igor Mol <igor.mol@makes.ai>
#
# The following program implements a Deep Belief Network (DBN) for time series 
# prediction using a combination of Convolutional Neural Network (CNN) and Long 
# Short-Term Memory (LSTM) layers. The DBNModel class is defined to encapsulate 
# the architecture and functionality of the model. The constructor initializes 
# parameters such as sequence length, CNN filters, CNN kernel size, and LSTM units. 
# The build_model method assembles the DBN architecture using a Sequential model
# from the Keras library. It includes a 1D Convolutional layer with ReLU activa-
# tion, a MaxPooling layer for down-sampling, an LSTM layer with ReLU activation,
# and a Dense output layer. The model is compiled with the Adam optimizer and mean
# squared error loss.
#     The main function, designated by the if name == "main": block, orchestrates 
# the overall process. It loads time series data from a CSV file, preprocesses it
# by scaling the values using Min-Max normalization, and structures the data into 
# input sequences and target values. The DBN model is instantiated, trained on the
# training set, and then used to predict the target values on the test set. The 
# predictions are denormalized to their original scale using the Min-Max scaler. 
# Two functions, print_table and plot_actual_vs_predicted, are defined to visually
# assess the model's performance. print_table generates a tabulated display of actual 
# versus predicted values, while plot_actual_vs_predicted produces a time series plot 
# comparing the actual and predicted values.
# In summary, the code demonstrates the construction and utilization of a DBN model
# for time series prediction, specifically tailored to sequences with a given length. 
# The integration of CNN and LSTM layers allows the model to capture both local 
# patterns through convolutional operations and long-term dependencies through 
# recurrent connections. The script culminates in the presentation of the model's
# predictions, enabling an evaluation of its effectiveness in capturing the underlying
# patterns within the time series data.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tabulate import tabulate
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense

# Define a class for the Deep Belief Network (DBN) model
class DBNModel:
    def __init__(self, sequence_length, cnn_filters=64, cnn_kernel_size=3, lstm_units=50):
        # Initialize model parameters
        self.sequence_length = sequence_length
        self.cnn_filters = cnn_filters
        self.cnn_kernel_size = cnn_kernel_size
        self.lstm_units = lstm_units
        # Build the DBN model upon instantiation
        self.model = self.build_model()

    # Define a method to build the DBN model
    def build_model(self):
        model = Sequential()
        # Add a 1D Convolutional layer with ReLU activation
        model.add(Conv1D(filters=self.cnn_filters, kernel_size=self.cnn_kernel_size, activation='relu',
                         input_shape=(self.sequence_length, 1)))
        # Add a MaxPooling layer for down-sampling
        model.add(MaxPooling1D(pool_size=2))
        # Add an LSTM layer with ReLU activation
        model.add(LSTM(self.lstm_units, activation='relu'))
        # Add a Dense output layer
        model.add(Dense(1))
        # Compile the model using Adam optimizer and mean squared error loss
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    # Define a method to train the DBN model
    def train(self, X_train, y_train, epochs=50, batch_size=32):
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2)

    # Define a method to make predictions using the trained model
    def predict(self, X_test):
        return self.model.predict(X_test)

# Define a function to print a table of actual versus predicted values
def print_table(actual, predicted):
    table = pd.DataFrame({
        'Actual': actual.flatten(),
        'Predicted': predicted.flatten()
    })
    print(tabulate(table, headers='keys', tablefmt='fancy_grid'))

# Define a function to plot the actual versus predicted time series
def plot_actual_vs_predicted(actual, predicted):
    plt.figure(figsize=(10, 6))
    plt.plot(actual, label='Actual')
    plt.plot(predicted, label='Predicted')
    plt.legend()
    plt.title('Actual vs. Predicted Time Series')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.show()

# Define the main function to execute the DBN model on time series data
def main():
    # Load and preprocess the time series data
    file_path = "/Users/igormol/Desktop/time_series_data.csv"  # Update with the actual file path
    time_series_df = pd.read_csv(file_path)
    time_series_df['Date'] = pd.to_datetime(time_series_df['Date'])
    values = time_series_df['Value'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    values_scaled = scaler.fit_transform(values)

    # Create input sequences and target values
    sequence_length = 10
    X, y = [], []
    for i in range(len(values_scaled) - sequence_length):
        X.append(values_scaled[i:i + sequence_length])
        y.append(values_scaled[i + sequence_length])

    X = np.array(X)
    y = np.array(y)

    # Split the data into training and testing sets
    train_size = int(len(values_scaled) * 0.8)
    X_train, X_test, y_train, y_test = X[:train_size], X[train_size:], y[:train_size], y[train_size:]

    # Create and train the DBN model
    dbn_model = DBNModel(sequence_length)
    dbn_model.train(X_train, y_train)

    # Make predictions on the test set
    y_pred = dbn_model.predict(X_test)

    # Denormalize the predictions and actual values
    y_pred_denormalized = scaler.inverse_transform(y_pred)
    y_test_denormalized = scaler.inverse_transform(y_test)

    # Print the table of predicted versus actual values
    print_table(y_test_denormalized, y_pred_denormalized)

    # Plot the actual versus predicted time series
    plot_actual_vs_predicted(y_test_denormalized, y_pred_denormalized)

# Execute the main function if the script is run directly
if __name__ == "__main__":
    main()
