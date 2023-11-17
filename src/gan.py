# Generative Adversarial Network for Time Series Analysis
#
# Igor Mol <igor.mol@makes.ai>
#
# This program uses a Generative Adversarial Network (GAN) for time series 
# analysis. The GAN consists of a generator and a discriminator. The generator 
# creates synthetic time series data, and the discriminator distinguishes 
# between real and generated data. The goal is to train the generator to produce 
# realistic time series data.
#     The program trains the GAN by generating synthetic time series and 
# updating the discriminator with real and generated data. It calculates and
# prints the discriminator and generator losses during training. The Mean 
# Squared Error (MSE) between actual and generated time series is also computed 
# and plotted to visualize the overall trend.
#     Finally, the program generates synthetic time series for the entire 
# dataset, creates a DataFrame with actual and generated data, prints the 
# results, and plots the actual and generated time series for comparison. The
# GAN learns to generate time series data that closely resembles the real data, 
# demonstrating its ability to capture underlying patterns in the time series.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Reshape, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class TimeSeriesGAN:
	
# Initialization:
# The class TimeSeriesGAN is initialized with a file path (file_path) pointing 
# to a CSV file with time series data and a latent dimension (latent_dim) for 
# the generator. Attributes include df (DataFrame for time series data), values
# (normalized time series values), scaler (MinMaxScaler for normalization), and
# models for the generator, discriminator, and GAN.

    def __init__(self, file_path, latent_dim=100):
        self.file_path = file_path
        self.latent_dim = latent_dim
        self.df = None
        self.values = None
        self.scaler = MinMaxScaler()
        self.generator = None
        self.discriminator = None
        self.gan = None

# load_data:
# - Reads CSV data into a DataFrame.
# - Converts the 'Date' column to datetime format, sorts the DataFrame by date.
# - Normalizes time series values using MinMaxScaler.

    def load_data(self):
        self.df = pd.read_csv(self.file_path)
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df = self.df.sort_values(by='Date')
        self.values = self.scaler.fit_transform(self.df['Value'].values.reshape(-1, 1))

# build_generator:
# - Constructs the generator model with dense layers, LeakyReLU activation, and 
# batch normalization.
# - The generator output has a single node with a sigmoid activation.
# - Compiles the model with binary crossentropy loss and the Adam optimizer.
        
    def build_generator(self):
        generator = Sequential()
        generator.add(Dense(128, input_dim=self.latent_dim))
        generator.add(LeakyReLU(alpha=0.2))
        generator.add(BatchNormalization(momentum=0.8))
        generator.add(Dense(1, activation='sigmoid'))
        generator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))
        self.generator = generator

# build_discriminator:
# - Constructs the discriminator model with a similar architecture to the generator.
# - Compiles the model with binary crossentropy loss, the Adam optimizer, and 
# accuracy as a metric.
        
    def build_discriminator(self):
        discriminator = Sequential()
        discriminator.add(Dense(128, input_dim=1))
        discriminator.add(LeakyReLU(alpha=0.2))
        discriminator.add(Dense(1, activation='sigmoid'))
        discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])
        self.discriminator = discriminator
        
# build_gan:
# - Builds the GAN model by combining the generator and discriminator.
# - Freezes discriminator weights during GAN training.
# - Compiles the GAN model with binary crossentropy loss and the Adam optimizer.
    
    def build_gan(self):
        self.discriminator.trainable = False
        gan_input = Input(shape=(self.latent_dim,))
        x = self.generator(gan_input)
        gan_output = self.discriminator(x)
        gan = Model(gan_input, gan_output)
        gan.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))
        self.gan = gan
        
# train_gan:
# - Trains the GAN for a specified number of epochs.
# - Generates synthetic time series, updates the discriminator with real and generated series.
# - Calculates and prints discriminator and generator losses at intervals.
# - Computes Mean Squared Error (MSE) between actual and generated time series.

    def train_gan(self, epochs=50, batch_size=64, sample_interval=1000):
        mse_history = []

        for epoch in range(epochs):
            noise = np.random.normal(0, 1, size=(batch_size, self.latent_dim))
            generated_series = self.generator.predict(noise)

            idx = np.random.randint(0, self.values.shape[0], batch_size)
            real_series = self.values[idx]

            labels_real = np.ones((batch_size, 1))
            labels_fake = np.zeros((batch_size, 1))

            d_loss_real = self.discriminator.train_on_batch(real_series, labels_real)
            d_loss_fake = self.discriminator.train_on_batch(generated_series, labels_fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            noise = np.random.normal(0, 1, size=(batch_size, self.latent_dim))
            labels_gan = np.ones((batch_size, 1))

            g_loss = self.gan.train_on_batch(noise, labels_gan)

            if epoch % sample_interval == 0:
                print(f"Epoch {epoch}, [D loss: {d_loss[0]} | D accuracy: {100 * d_loss[1]}] [G loss: {g_loss}]")

            # Calculate MSE and store for plotting
            generated_series = self.generate_synthetic_series(num_samples=self.values.shape[0])
            mse = mean_squared_error(self.values, generated_series)
            mse_history.append(mse)

        # Plot the overall trend of MSE
        self.plot_mse_trend(mse_history)

# generate_synthetic_series:
# Generates synthetic time series using the trained generator.

    def generate_synthetic_series(self, num_samples):
        noise = np.random.normal(0, 1, size=(num_samples, self.latent_dim))
        generated_series = self.generator.predict(noise)
        return self.scaler.inverse_transform(generated_series)
        
# plot_mse_trend method:
# Plots the overall trend of Mean Squared Error (MSE) during GAN training.

    def plot_mse_trend(self, mse_history):
        plt.figure(figsize=(8, 4))
        plt.plot(range(1, len(mse_history) + 1), mse_history, marker='o')
        plt.title('Overall Trend of Mean Squared Error (MSE)')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.grid(True)
        plt.show()

# visualize_results:
# - Generates synthetic time series for the entire dataset.
# - Creates a DataFrame with actual and generated time series.
# - Prints results in a formatted table.
# - Plots the actual and generated time series.
    
    def visualize_results(self):
        # Generate synthetic time series
        generated_series = self.generate_synthetic_series(num_samples=self.values.shape[0])

        # Create a DataFrame for the results
        results = pd.DataFrame({'Date': self.df['Date'], 'Actual': self.df['Value'], 'Generated': generated_series.flatten()})

        # Print the results in a formatted table
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        print(results)
        pd.reset_option('display.max_rows')
        pd.reset_option('display.max_columns')

        # Plot the actual and generated time series
        plt.figure(figsize=(12, 6))
        plt.plot(self.df['Date'], self.df['Value'], label='Actual', marker='o')
        plt.plot(self.df['Date'], generated_series, label='Generated', marker='o')
        plt.title('Generative Adversarial Network Time Series Generation')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.show()
     
# Main function:   
# - Creates an instance of TimeSeriesGAN.
# - Loads data, builds generator, discriminator, and GAN models.
# - Trains the GAN and visualizes the results.

def main():
    gan_model = TimeSeriesGAN(file_path="/Users/igormol/Desktop/time_series_data.csv")
    gan_model.load_data()
    gan_model.build_generator()
    gan_model.build_discriminator()
    gan_model.build_gan()
    gan_model.train_gan()
    gan_model.visualize_results()

if __name__ == "__main__":
    main()
