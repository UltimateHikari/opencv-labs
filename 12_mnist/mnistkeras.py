# source: https://www.tensorflow.org/tutorials/keras/save_and_load
import os
import sys

import tensorflow as tf
from tensorflow import keras

print(tf.version.VERSION)

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

SIZE = int(sys.argv[1])

train_labels = train_labels[:SIZE]
test_labels = test_labels[:SIZE]

train_images = train_images[:SIZE].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:SIZE].reshape(-1, 28 * 28) / 255.0

# Define a simple sequential model
def create_model():
  model = tf.keras.Sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10)
  ])

  model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

  return model

# Create a basic model instance
model = create_model()

# Display the model's architecture
model.summary()

model.fit(train_images, train_labels, epochs=5)

# Save the entire model to a HDF5 file.
# The '.h5' extension indicates that the model should be saved to HDF5.
model.save('./models/mnistdense.h5')
