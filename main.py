import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

# Generate dummy data
x_train = np.random.rand(1000, 100)
y_train = np.random.randint(2, size=(1000, 100))

# Define custom loss function
# this will define both work and replication
def custom_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

# Create the neural network model with explicit input layer
model = Sequential([
    Input(shape=(200,)),                              # Explicit input layer
    Dense(32, activation='relu'),                    # First hidden layer
    Dense(16, activation='relu'),                    # Second hidden layer
    Dense(200, activation='sigmoid')                 # Output layer with 100 parameters
])

# Compile the model with custom loss function
model.compile(optimizer='adam', loss=custom_loss, metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=32)

# Evaluate the model
loss, accuracy = model.evaluate(x_train, y_train)
print(f'Accuracy: {accuracy:.2f}')

