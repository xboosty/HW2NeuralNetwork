# HW2NeuralNetwork

# Install necessary packages (usually done in a Jupyter notebook or Python script)
!pip install matplotlib
!pip install scikit-learn

# Import necessary libraries
import pandas as pd
import ast
import numpy as np
from tensorflow import keras
import tensorflow as tf

# Mount Google Drive if it's not already mounted
from google.colab import drive
drive.mount('/content/drive')

# Load the dataset from Google Drive
dataset_path = '/content/drive/My Drive/compatibility_dataset/compatibility_dataset_len76.csv'

# Read the CSV file and process the lists
df = pd.read_csv(dataset_path)

# Flatten and convert lists into NumPy arrays
def flatten_and_convert(row):
    try:
        offers_A = ast.literal_eval(row['offers_A'])
        demands_A = ast.literal_eval(row['demands_A'])
        offers_B = ast.literal_eval(row['offers_B'])
        demands_B = ast.literal_eval(row['demands_B'])

        flat_data = np.concatenate([offers_A, demands_A, offers_B, demands_B])
        return flat_data.astype(np.int32)
    except:
        return None

# Apply the function to each row of the DataFrame
X = df.apply(flatten_and_convert, axis=1).dropna().values
y = df['compatibility_percent'].dropna().values

# Create a sequential model
model = keras.Sequential()

# Input layers
model.add(keras.layers.Input(shape=(152,)))

# Processing layers with regularization
model.add(keras.layers.Dense(128, activation='gelu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.1))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(32, activation='relu'))
model.add(keras.layers.Dense(16, activation='relu'))

# Similarity layer
model.add(keras.layers.Dense(1, activation='linear'))

# Output layer to limit the output range between 0 and 100
model.add(keras.layers.Lambda(lambda x: tf.clip_by_value(x, 0, 100)))

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# Model summary
model.summary()

# Train the model
# Convert NumPy arrays to TensorFlow tensors
X_tf = tf.constant(X.tolist())
y_tf = tf.constant(y)

# Then, use these tensors in the model.fit function
history = model.fit(X_tf, y_tf, epochs=50, validation_split=0.2)
