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
# from tensorflow.keras.datasets import mnist
import psutil
import GPUtil
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
# def convert_to_tflite(model, filename):
#    converter = tf.lite.TFLiteConverter.from_keras_model(model)
#    converter.optimizations = [tf.lite.Optimize.DEFAULT]
#   tflite_model = converter.convert()

#   with open(filename, 'wb') as f:
#       f.write(tflite_model)

# Convert both models
# convert_to_tflite(qat_simple_cnn_model, 'simple_cnn_qat_int8_mnist.tflite')
# convert_to_tflite(qat_mobilenetv2_model, 'mobilenetv2_qat_int8_mnist.tflite')

# print("Models converted and saved.")

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

from tensorflow_model_optimization.python.core.keras.compat import keras


def infer(interpreter, input_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Prepare input data
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get prediction and confidence
    output = interpreter.get_tensor(output_details[0]['index'])
    confidence = np.max(output)
    # predicted_class = np.argmax(output)
    return confidence


def get_resource_utilization():
    cpu_utilization = psutil.cpu_percent(interval=1)
    memory_utilization = psutil.virtual_memory().percent
    return cpu_utilization, memory_utilization


def switch_model(input_data, mobilenet_model_path, resnet_model_path, confidence_threshold, resource_limit):
    cpu_util, memory_util = get_resource_utilization()
    print(f"CPU: {cpu_util}%, Memory: {memory_util}%")
    mobilenet_interpreter = tflite.Interpreter(model_path=mobilenet_model_path)
    resnet_interpreter = tflite.Interpreter(model_path=resnet_model_path)

    mobilenet_interpreter.allocate_tensors()
    resnet_interpreter.allocate_tensors()

    conf_threshold = confidence_threshold
    cpu_resource_limit = resource_limit

    # Infer with Resnet first
    confidence = infer(resnet_interpreter, input_data)
    print(f"Resnet Confidence: {confidence}")

    # If confidence is low and resources allow, switch to ResNet
    if confidence < conf_threshold and cpu_util < cpu_resource_limit:
        print("Switching to MobileNet for better accuracy...")
        confidence = infer(mobilenet_interpreter, input_data)
        print(f"MobileNet Confidence: {confidence}")

    return confidence


def switch_model_infer(input_data, running_model_path):
    mobilenet_interpreter = tflite.Interpreter(model_path=running_model_path)
    # resnet_interpreter = tflite.Interpreter(model_path=resnet_model_path)

    mobilenet_interpreter.allocate_tensors()
    # resnet_interpreter.allocate_tensors()

    # conf_threshold = confidence_threshold
    # cpu_resource_limit = resource_limit

    # Infer with Resnet first
    confidence = infer(mobilenet_interpreter, input_data)
    print(f"Confidence: {confidence}")

    # If confidence is low and resources allow, switch to ResNet
    # if confidence < conf_threshold and cpu_util < cpu_resource_limit:
    #    print("Switching to MobileNet for better accuracy...")
    #    confidence, predicted_class = infer(mobilenet_interpreter, input_data)
    #    print(f"MobileNet Confidence: {confidence}")

    return confidence


def get_gpu_usage():
    """Get current GPU memory usage."""
    gpus = GPUtil.getGPUs()
    if gpus:
        return max([gpu.memoryUtil for gpu in gpus]) * 100  # Get memory utilization in percentage
    return 60  # Return None if no GPU is available


# Step 3: Resource Cost Function with Confidence and Task Complexity
def compute_resource_cost(accuracy, confidence, cpu_usage, gpu_usage, memory_usage, task_complexity, weights):
    """
    Compute the resource cost function with task complexity and inference confidence.
    
    :param accuracy: Model accuracy.
    :param confidence: Model's inference confidence.
    :param cpu_usage: Current CPU usage.
    :param gpu_usage: Current GPU usage.
    :param memory_usage: Current memory usage.
    :param task_complexity: The complexity of the current task.
    :param weights: A dictionary with weight values for each component.
    :return: Computed cost.
    """
    alpha = weights['accuracy']
    beta = weights['cpu']
    gamma = weights['gpu']
    delta = weights['memory']
    epsilon = weights['confidence']
    zeta = weights['complexity']

    # Cost function: lower cost is better
    cost = (alpha * (1 - accuracy) +
            epsilon * (1 - confidence) +
            beta * cpu_usage +
            gamma * gpu_usage +
            delta * memory_usage +
            zeta * task_complexity)

    return cost


# dynamic_model = DynamicModelSwitching("mobilenet_qat_cv_int8_qat.tflite", "resnet_qat_cv_int8_qat.tflite")

# Step 4: Dynamic model switching with confidence and task complexity
def dynamic_model_switching_with_complexity(num_epochs=10, resource_limits=None, weights=None, task_complexity=1):
    """
    Perform dynamic model switching based on resource cost, inference confidence, and task complexity.
    
    :param resource_limits: Dictionary containing resource thresholds.
    :param weights: Weights for accuracy, confidence, CPU, GPU, memory, and complexity in the cost function.
    :param task_complexity: Complexity of the task (simple=1, complex=5).
    """
    if resource_limits is None:
        resource_limits = {'cpu': 70, 'gpu': 80, 'memory': 75}  # Default limits

    if weights is None:
        weights = {'accuracy': 0.4, 'confidence': 0.3, 'cpu': 0.1, 'gpu': 0.1, 'memory': 0.1,
                   'complexity': 0.2}  # Default weights

    for epoch in range(num_epochs):
        cost = 0.0

        # Test on MNIST-data
        for i in range(5):
            cpu_usage, memory_usage = get_resource_utilization()
            gpu_usage = get_gpu_usage()
            input_image = np.expand_dims(x_test[i], axis=0).astype(np.float32)
            print(f"Running inference on sample {i + 1}")
            for model_name in ("mobilenet_qat_cv_int8_qat.tflite", "resnet_qat_cv_int8_qat.tflite"):
                accuracy = 0.6
                confidence = switch_model_infer(input_image, model_name)
                print(f"Input Image Confidence: {confidence}")
                cost = compute_resource_cost(accuracy, confidence, cpu_usage, gpu_usage, memory_usage, task_complexity,
                                             weights)
                print(f"Model: {model_name}, Cost: {cost:.4f}")
                # Choose the model with the lowest cost
                if model_name == "mobilenet_qat_cv_int8_qat.tflite" or model_name == "resnet_qat_cv_int8_qat.tflite":  # Adjust conditions based on actual scenario
                    switch_model(input_image, "mobilenet_qat_cv_int8_qat.tflite", "resnet_qat_cv_int8_qat.tflite",
                                 confidence, cpu_usage)

                # cpu_usage,memory_usage = get_resource_utilization()
                # gpu_usage = get_gpu_usage()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Cost: {cost}")

        # Validate the model and update accuracy and confidence
        # validate_model(model_manager, val_loader)

    print("Dynamic switching complete")


if __name__ == "__main__":
    dynamic_model_switching_with_complexity(num_epochs=10, resource_limits=None, weights=None, task_complexity=1)
