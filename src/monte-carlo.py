# Monte Carlo Method to Detect Time Series Anomalies
#
# Igor Mol <igor.mol@makes.ai>
#
# In this program, the Monte Carlo method is applied to analyze time series 
# data for detecting and handling anomalies. The code begins by creating a class 
# called MonteCarloAnomalyDetection with key parameters like the file path to a 
# CSV containing time series data, a threshold for anomaly detection, and the 
# number of simulations for anomaly replacement. After loading and normalizing 
# the time series data, the code uses Z-scores to identify anomalies. Z-scores 
# measure how far each data point deviates from the mean, and if this deviation 
# exceeds a set threshold, the point is marked as an anomaly. 
#     The anomalies are then replaced using Monte Carlo simulations. For each 
# anomaly, simulated values are generated from a normal distribution based on 
# the mean and standard deviation of the data, and the anomaly is replaced with 
# the mean of these simulated values.
#     The main function of the code executes the entire anomaly detection 
# process, displaying the results in a formatted table that includes the date, 
# original values, replaced values, and an indicator for anomalies. Additional-
# ly, the program creates a time series plot highlighting the original and r
# eplaced values, emphasizing the detected anomalies in red. The Monte Carlo 
# technique utilized in this code is a statistical approach that leverages ran-
# domness through simulations. By replacing anomalies with simulated values 
# based on the data's statistical properties, the Monte Carlo method offers a 
# flexible and effective way to handle outliers and uncertainties in time series 
# data, showcasing its versatility in practical applications.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# The `MonteCarloAnomalyDetection' class is initiated with essential parameters: 
# the path to a CSV file containing time series data, a threshold for anomaly 
# detection, and the number of simulations for anomaly replacement. Default va-
# lues for the threshold and number of simulations are set at 1.5 and 1000, res-
# pectively. The class holds attributes such as:
# - file_path, 
# - threshold, 
# - num_simulations, 
# - df (DataFrame for time series data), 
# - anomalies, and 
# - replaced_values.

class MonteCarloAnomalyDetection:

    def __init__(self, file_path, threshold=1.5, num_simulations=1000):
        self.file_path = file_path
        self.threshold = threshold
        self.num_simulations = num_simulations
        self.df = None
        self.anomalies = None
        self.replaced_values = None

    def load_data(self):
        self.df = pd.read_csv(self.file_path)
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df = self.df.sort_values(by='Date')
        
# detect_anomalies:
# - Detects anomalies in the time series based on Z-scores.
# - Z-scores are calculated as the difference between data and the mean, divided 
# by the standard deviation.
# - Anomalies are identified if the Z-score exceeds the defined threshold.

    def detect_anomalies(self, data):
        mean_val = np.mean(data)
        std_val = np.std(data)
        z_scores = np.abs((data - mean_val) / std_val)
        anomalies = z_scores > self.threshold
        return anomalies

# replace_anomalies:
# - Replaces anomalies in the time series using Monte Carlo simulations.
# - For each data point identified as an anomaly, simulated values are generated 
# from a normal distribution based on the mean and standard deviation of the data.
# - The anomaly value is replaced by the mean of the simulated values.
    
    def replace_anomalies(self, data):
        replaced_data = data.copy()
        for i in range(len(data)):
            if self.anomalies[i]:
                simulation_values = np.random.normal(np.mean(data), np.std(data), self.num_simulations)
                replaced_data[i] = np.mean(simulation_values)
        return replaced_data

# run_anomaly_detection:
# - Executes the anomaly detection process.
# - Calls load_data, detect_anomalies, and replace_anomalies to populate anoma-
# lies and replaced_values.
    
    def run_anomaly_detection(self):
        self.load_data()
        self.anomalies = self.detect_anomalies(self.df['Value'])
        self.replaced_values = self.replace_anomalies(self.df['Value'])

# display_results:
# - Displays results in a formatted table using a DataFrame.
# - Displays 'Date', 'Original' time series values, 'Replaced' values after ano-
# maly replacement, and 'Anomaly' flag.
    
    def display_results(self):
        results = pd.DataFrame({'Date': self.df['Date'], 'Original': self.df['Value'],
                                'Replaced': self.replaced_values, 'Anomaly': self.anomalies})
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        print(results)
        pd.reset_option('display.max_rows')
        pd.reset_option('display.max_columns')

# plot_time_series:
# - Plots the original time series, the series with replaced anomalies, and 
# highlights anomaly points.
# - Uses matplotlib to create a time series plot.
    
    def plot_time_series(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.df['Date'], self.df['Value'], label='Original', marker='o')
        plt.plot(self.df['Date'], self.replaced_values, label='Replaced', marker='o')
        plt.scatter(self.df['Date'][self.anomalies], self.df['Value'][self.anomalies], color='red', label='Anomalies')
        plt.title('Monte Carlo Anomaly Detection and Replacement')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.show()

def main():
    anomaly_detector = MonteCarloAnomalyDetection(file_path="/Users/igormol/Desktop/time_series_data.csv")
    anomaly_detector.run_anomaly_detection()
    anomaly_detector.display_results()
    anomaly_detector.plot_time_series()

if __name__ == "__main__":
    main()
