import tempfile
import os

import tensorflow as tf
import numpy as np
import tensorflow_model_optimization as tfmot
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
#from tensorflow.keras.datasets import mnist
import psutil
import numpy as np
import tensorflow.lite as tflite
import time
from tensorflow_model_optimization.python.core.keras.compat import keras


# Load and preprocess MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Resize MNIST to 32x32 for MobileNetV2 input
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)
x_train = tf.image.resize(x_train, [32, 32])
x_test = tf.image.resize(x_test, [32, 32])

# Normalize the input data
x_train, x_test = x_train / 255.0, x_test / 255.0

# One-hot encode the labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


# Define a simple CNN model for less complex tasks
def create_simple_cnn():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model

# Define MobileNetV2 model for more complex tasks
def create_mobilenetv2_model():
    base_model = MobileNetV2(input_shape=(32, 32, 1), include_top=False, weights=None)
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    output = layers.Dense(10, activation='softmax')(x)
    model = models.Model(inputs=base_model.input, outputs=output)
    return model

# Apply Quantization-Aware Training (QAT) to a model
def apply_qat(model):
    quantize_model = tfmot.quantization.keras.quantize_model
    qat_model = quantize_model(model)
    return qat_model

# Create models
simple_cnn_model = create_simple_cnn()
mobilenetv2_model = create_mobilenetv2_model()

# Apply QAT to both models
qat_simple_cnn_model = apply_qat(simple_cnn_model)
qat_mobilenetv2_model = apply_qat(mobilenetv2_model)

# Compile both models
qat_simple_cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
qat_mobilenetv2_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train both models
qat_simple_cnn_model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
qat_mobilenetv2_model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Convert models to TensorFlow Lite with INT8 quantization
#def convert_to_tflite(model, filename):
#    converter = tf.lite.TFLiteConverter.from_keras_model(model)
#    converter.optimizations = [tf.lite.Optimize.DEFAULT]
 #   tflite_model = converter.convert()

 #   with open(filename, 'wb') as f:
 #       f.write(tflite_model)

# Convert both models
#convert_to_tflite(qat_simple_cnn_model, 'simple_cnn_qat_int8_mnist.tflite')
#convert_to_tflite(qat_mobilenetv2_model, 'mobilenetv2_qat_int8_mnist.tflite')

#print("Models converted and saved.")

def convert_qat_to_tflite(qat_model, model_name):
    converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    # Save the converted model
    with open(f"{model_name}_cv_int8_qat.tflite", "wb") as f:
        f.write(tflite_model)
    print(f"Model saved: {model_name}_cv_int8_qat.tflite")


 # Convert both models
convert_qat_to_tflite(qat_simple_cnn_model, "resnet_qat")
convert_qat_to_tflite(qat_mobilenetv2_model, "mobilenet_qat")

class DynamicModelSwitching:
    def __init__(self, mobilenet_model_path, resnet_model_path, confidence_threshold=0.7, resource_limit=80):
        # Load TensorFlow Lite interpreters
        self.mobilenet_interpreter = tflite.Interpreter(model_path=mobilenet_model_path)
        self.resnet_interpreter = tflite.Interpreter(model_path=resnet_model_path)
        self.mobilenet_interpreter.allocate_tensors()
        self.resnet_interpreter.allocate_tensors()

        self.confidence_threshold = confidence_threshold
        self.resource_limit = resource_limit

    def infer(self, interpreter, input_data):
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Prepare input data
        interpreter.set_tensor(input_details[0]['index'], input_data)

        # Run inference
        interpreter.invoke()

        # Get prediction and confidence
        output = interpreter.get_tensor(output_details[0]['index'])
        confidence = np.max(output)
        predicted_class = np.argmax(output)
        return confidence, predicted_class

    def get_resource_utilization(self):
        cpu_utilization = psutil.cpu_percent(interval=1)
        memory_utilization = psutil.virtual_memory().percent
        return cpu_utilization, memory_utilization

    def switch_model(self, input_data):
        cpu_util, memory_util = self.get_resource_utilization()
        print(f"CPU: {cpu_util}%, Memory: {memory_util}%")

        # Infer with Resnet first me Lio
        confidence, predicted_class = self.infer(self.resnet_interpreter, input_data)
        print(f"ReSNet Confidence: {confidence}")

        # If confidence is low and resources allow, switch to ResNet
        if confidence < self.confidence_threshold and cpu_util < self.resource_limit:
            print("Switching to MobileNet for better accuracy...")
            confidence, predicted_class = self.infer(self.mobilenet_interpreter, input_data)
            print(f"MobileNet Confidence: {confidence}")
        #Infer with end Lio
        # Infer with MobileNet first
        #confidence, predicted_class = self.infer(self.mobilenet_interpreter, input_data)
        #print(f"MobileNet Confidence: {confidence}")

        # If confidence is low and resources allow, switch to ResNet
        #if confidence < self.confidence_threshold and cpu_util < self.resource_limit:
        #    print("Switching to ResNet for better accuracy...")
        #    confidence, predicted_class = self.infer(self.resnet_interpreter, input_data)
        #    print(f"ResNet Confidence: {confidence}")

        return confidence, predicted_class


def resource_cost_function(inference_time, cpu_util, memory_util):
    """Compute resource cost as a weighted sum of time, CPU, and memory usage."""
    return inference_time + 0.5 * cpu_util + 0.3 * memory_util

def measure_inference_time(interpreter, input_data):
    """Measure inference time for the model."""
    start_time = time.time()
    interpreter.invoke()
    end_time = time.time()
    return end_time - start_time

dynamic_model = DynamicModelSwitching("mobilenet_qat_cv_int8_qat.tflite", "resnet_qat_cv_int8_qat.tflite")

# Test on MNIST-data
for i in range(10):
    input_image = np.expand_dims(x_test[i], axis=0).astype(np.float32)
    print(f"Running inference on sample {i + 1}")
    confidence, predicted_class = dynamic_model.switch_model(input_image)
    print(f"Final Confidence: {confidence}, Predicted Class: {predicted_class}")


