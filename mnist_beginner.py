'''
https://www.tensorflow.org/tutorials/quickstart/beginner

Dataset Notes:
 * mnist.load_data downloads the MNIST data set from the web, unless
   cached as ~/.keras/datasets/mnist.npz.
 * mnist.npz is a zip file.  It can be loaded with numpy.load() which
   will return a dict-like object with four elements: x_train, y_train,
   x_test, and y_test.
 * Each of the files in the zip file can be loaded using numpy.load,
   which will each return an ndarray.
 * Dimensions:
   x_test: 60000 x 28 x 28   (60,000 28x28 images)
   y_test: 60000 x 1
'''

# Following __future__ imports note needed for python 3.7
# from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

mnist = tf.keras.datasets.mnist

# Load and prepare the MNIST dataset.
(x_train, y_train), (x_test, y_test) = mnist.load_data()
#  Convert the samples from integers to floating-point numbers in the range [0, 1.0]
x_train, x_test = x_train / 255.0, x_test / 255.0

# for reproducable results, set the random seed prior to building the model
tf.random.set_seed(4242)

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

# Running model.fit() again continues the training from where we left off
model.fit(x_train, y_train, epochs=4)

# Re-evaluate
model.evaluate(x_test,  y_test, verbose=2)

'''
$ python3 mnist_beginner.py
2019-10-07 16:51:36.109172: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2019-10-07 16:51:36.126874: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x7ffb1091db00 executing computations on platform Host. Devices:
2019-10-07 16:51:36.126896: I tensorflow/compiler/xla/service/service.cc:175]   StreamExecutor device (0): Host, Default Version
Train on 60000 samples
Epoch 1/5
60000/60000 [==============================] - 3s 51us/sample - loss: 0.2950 - accuracy: 0.9148
Epoch 2/5
60000/60000 [==============================] - 3s 42us/sample - loss: 0.1443 - accuracy: 0.9575
Epoch 3/5
60000/60000 [==============================] - 3s 42us/sample - loss: 0.1084 - accuracy: 0.9665
Epoch 4/5
60000/60000 [==============================] - 3s 42us/sample - loss: 0.0898 - accuracy: 0.9725
Epoch 5/5
60000/60000 [==============================] - 3s 43us/sample - loss: 0.0768 - accuracy: 0.9762
10000/1 - 0s - loss: 0.0392 - accuracy: 0.9754

Train on 60000 samples
Epoch 1/4
60000/60000 [==============================] - 3s 45us/sample - loss: 0.0673 - accuracy: 0.9784
Epoch 2/4
60000/60000 [==============================] - 3s 43us/sample - loss: 0.0596 - accuracy: 0.9815
Epoch 3/4
60000/60000 [==============================] - 3s 43us/sample - loss: 0.0548 - accuracy: 0.9822
Epoch 4/4
60000/60000 [==============================] - 3s 43us/sample - loss: 0.0481 - accuracy: 0.9845
10000/1 - 0s - loss: 0.0344 - accuracy: 0.9802
'''
