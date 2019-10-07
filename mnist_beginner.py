from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

mnist = tf.keras.datasets.mnist

# Load and prepare the MNIST dataset.
(x_train, y_train), (x_test, y_test) = mnist.load_data()
#  Convert the samples from integers to floating-point numbers in the range [0, 1.0]
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build the tf.keras.Sequential model by stacking layers.
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

# Choose an optimizer and loss function for training.
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5)

# Evaluate the model
model.evaluate(x_test,  y_test, verbose=2)

'''
$ python3 mnist_beginner.py
2019-10-07 16:51:36.109172: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2019-10-07 16:51:36.126874: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x7ffb1091db00 executing computations on platform Host. Devices:
2019-10-07 16:51:36.126896: I tensorflow/compiler/xla/service/service.cc:175]   StreamExecutor device (0): Host, Default Version
Train on 60000 samples
Epoch 1/5
60000/60000 [==============================] - 3s 50us/sample - loss: 0.2925 - accuracy: 0.9144
Epoch 2/5
60000/60000 [==============================] - 3s 42us/sample - loss: 0.1390 - accuracy: 0.9585
Epoch 3/5
60000/60000 [==============================] - 3s 42us/sample - loss: 0.1044 - accuracy: 0.9678
Epoch 4/5
60000/60000 [==============================] - 3s 42us/sample - loss: 0.0875 - accuracy: 0.9724
Epoch 5/5
60000/60000 [==============================] - 2s 42us/sample - loss: 0.0749 - accuracy: 0.9769
10000/1 - 0s - loss: 0.0401 - accuracy: 0.9763
'''
