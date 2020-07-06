#!/usr/bin/python
import tensorflow as tf # imports the main library to use in machine learning
mn = tf.keras.datasets.mnist # Import the mnist hand-written-number dataset, which is what I will practice using
# tensorflow on

(x_train, y_train),(x_test, y_test) = mn.load_data() # Load the mnist training and testing data into
# their 4 arrays. The training data prepares the machine learning model for the test data set.
x_train, x_test = x_train / 255.0, x_test / 255.0 # Convert the testing and training data to floating
# point values from 0-1, as the values originally range from 0 to 255 in a 28x28 array

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5) # begins the training of the model over the course of 5 iterations
# (epochs = iterations of training)
model.evaluate(x_test, y_test) #Evaluates the model against the test cases
