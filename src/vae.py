# Variational Autoencoder for Time Series Analysis
#
# Igor Mol <igor.mol@makes.ai> 
#
# The code that follows implements a Variational Autoencoder (VAE) for recons-
# tructing time series data. This VAE architecture consists of an encoder, a 
# reparameterization trick, and a decoder. The encoder processes the input time 
# series, transforming it into a condensed representation in a latent space. 
#     The reparameterization trick introduces randomness to navigate uncertainty 
# in this latent space. The decoder then reconstructs the time series from this 
# latent representation. The VAE's performance is evaluated through a loss fun-
# ction, comprising cross-entropy loss and Kullback-Leibler divergence. During 
# training, the model refines its understanding of the time series through 
# stochastic gradient descent. The encoded series reveals latent insights, while 
# the reconstructed series is brought back to the original scale. This process 
# exemplifies the VAE's ability to understand and reconstruct time series data, 
# bridging classical principles with modern neural network techniques.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras import backend as K
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
from scipy.stats import norm

# Define a class for the Variational Autoencoder (VAE)
class VAE:
    def __init__(self, original_dim, intermediate_dim, latent_dim):
        # Initialize VAE with dimensions for input, intermediate layer, and latent space
        self.original_dim = original_dim
        self.intermediate_dim = intermediate_dim
        self.latent_dim = latent_dim
        # Build the VAE model
        self.model = self.build_model()

    def build_model(self):
        # Encoder architecture
        inputs = Input(shape=(self.original_dim,), name='encoder_input')
        h = Dense(self.intermediate_dim, activation='relu')(inputs)
        z_mean = Dense(self.latent_dim, name='z_mean')(h)
        z_log_var = Dense(self.latent_dim, name='z_log_var')(h)

        # Reparameterization trick
        def sampling(args):
            z_mean, z_log_var = args
            batch = K.shape(z_mean)[0]
            dim = K.int_shape(z_mean)[1]
            epsilon = K.random_normal(shape=(batch, dim))
            return z_mean + K.exp(0.5 * z_log_var) * epsilon

        z = Lambda(sampling, output_shape=(self.latent_dim,), name='z')([z_mean, z_log_var])

        # Decoder architecture
        decoder_h = Dense(self.intermediate_dim, activation='relu')
        decoder_mean = Dense(self.original_dim, activation='sigmoid')
        h_decoded = decoder_h(z)
        x_decoded_mean = decoder_mean(h_decoded)

        # Overall VAE model
        vae = Model(inputs, x_decoded_mean)

        # VAE loss and custom layer
        xent_loss = self.original_dim * binary_crossentropy(inputs, x_decoded_mean)
        kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        vae_loss = K.mean(xent_loss + kl_loss)

        # Add the loss to the model and compile
        vae.add_loss(vae_loss)
        vae.compile(optimizer=Adam())
        
        return vae

    def train(self, data, epochs=50, batch_size=32, validation_split=0.1):
        # Train the VAE model on the provided data
        self.model.fit(data, epochs=epochs, batch_size=batch_size, validation_split=validation_split)

    def predict(self, data):
        # Use the trained model to predict reconstructed data
        return self.model.predict(data)

# Function to normalize data using Min-Max scaling
def normalize_data(data):
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(data.reshape(-1, 1))
    return normalized_data, scaler

# Function to denormalize data
def denormalize_data(normalized_data, scaler):
    return scaler.inverse_transform(normalized_data).flatten()

# Main function
def main():
    # Load the time-series data from a CSV file
    file_path = "/Users/igormol/Desktop/time_series_data.csv"
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date')

    # Normalize the 'Value' column
    values = df['Value'].values
    normalized_values, scaler = normalize_data(values)

    # Create a VAE model
    # "In our sleep, pain which cannot forget falls drop by drop upon the heart until, in our own despair, against our will, comes wisdom through the awful grace of God." - Aeschylus
    vae_model = VAE(original_dim=1, intermediate_dim=64, latent_dim=2)

    # Train the VAE on the normalized time series
    X_train = normalized_values.reshape(-1, 1)
    vae_model.train(X_train)

    # Encode and decode the time series
    encoded_series = vae_model.predict(X_train)
    decoded_series = denormalize_data(encoded_series, scaler)

    # Create a DataFrame for the results
    results = pd.DataFrame({'Date': df['Date'], 'Actual': values, 'Reconstructed': decoded_series})

    # Print the results in a formatted table
    print(results)

    # Plot the actual and reconstructed time series
    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], values, label='Actual', marker='o')
    plt.plot(df['Date'], decoded_series, label='Reconstructed', marker='o')
    plt.title('Variational Autoencoder Time Series Reconstruction')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

# Run the main function if the script is executed directly
if __name__ == "__main__":
    main()
