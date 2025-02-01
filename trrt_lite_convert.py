import tensorflow as tf
import psutil
import time
import numpy as np
import tensorflow.lite as tflite
from tensorflow.keras.applications import ResNet50, MobileNetV2
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow_model_optimization.quantization.keras import quantize_model
import tensorflow_model_optimization as tfmot

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
#x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize data


#Define a simple model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile and train a simple model

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=64, epochs = 10, validation_data=(x_test, y_test))

#convert to TFLIte

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("cifar10_model_trrt.tflite", "wb") as f:
    f.write(tflite_model)
