import tensorflow as tf
import psutil
import numpy as np
import tensorflow.lite as tflite
from tensorflow.keras.applications import ResNet50, MobileNetV2
from tensorflow.keras.datasets import cifar10


# Load CIFAR-10 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize data

# Train ResNet and MobileNet models
def create_resnet_model():
    model = ResNet50(weights=None, input_shape=(32, 32, 3), classes=10)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def create_mobilenet_model():
    model = MobileNetV2(weights=None, input_shape=(32, 32, 3), classes=10)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

resnet_model = create_resnet_model()
mobilenet_model = create_mobilenet_model()

# Train models
resnet_model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
mobilenet_model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

def convert_to_tflite(model, model_name):
    """Convert Keras model to TensorFlow Lite INT8 model"""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    def representative_dataset_gen():
        for input_value in tf.data.Dataset.from_tensor_slices(x_train).batch(1).take(100):
            yield [tf.cast(input_value, tf.float32)]
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    tflite_model = converter.convert()
    
    # Save the model
    with open(f"{model_name}_int8.tflite", "wb") as f:
        f.write(tflite_model)
    print(f"INT8 model saved: {model_name}_int8.tflite")

# Convert ResNet and MobileNet to INT8
convert_to_tflite(resnet_model, "resnet")
convert_to_tflite(mobilenet_model, "mobilenet")


def get_resource_utilization():
    """Return CPU and memory utilization percentages"""
    cpu_utilization = psutil.cpu_percent(interval=1)
    memory_utilization = psutil.virtual_memory().percent
    return cpu_utilization, memory_utilization


class DynamicModelSwitching:
    def __init__(self, mobilenet_model_path, resnet_model_path, confidence_threshold=0.7, resource_limit=80):
        # Load TensorFlow Lite models
        self.mobilenet_interpreter = tflite.Interpreter(model_path=mobilenet_model_path)
        self.resnet_interpreter = tflite.Interpreter(model_path=resnet_model_path)
        self.confidence_threshold = confidence_threshold
        self.resource_limit = resource_limit  # Max CPU/memory usage allowed

        self.mobilenet_interpreter.allocate_tensors()
        self.resnet_interpreter.allocate_tensors()

    def infer(self, interpreter, input_data):
        """Run inference on the provided interpreter"""
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Set input data
        interpreter.set_tensor(input_details[0]['index'], input_data)

        # Run inference
        interpreter.invoke()

        # Get output (confidence score)
        output = interpreter.get_tensor(output_details[0]['index'])
        confidence = np.max(output)
        predicted_class = np.argmax(output)
        return confidence, predicted_class

    def decide_model(self, input_data):
        """Switch between models based on confidence and resource availability"""
        # Check resource usage
        cpu_util, memory_util = get_resource_utilization()
        print(f"CPU: {cpu_util}%, Memory: {memory_util}%")
        
        # Use MobileNet first (lightweight model)
        confidence, pred_class = self.infer(self.mobilenet_interpreter, input_data)
        print(f"MobileNet Confidence: {confidence}")

        # If confidence is low and resources allow, switch to ResNet
        if confidence < self.confidence_threshold and cpu_util < self.resource_limit:
            print("Switching to ResNet for better accuracy...")
            confidence, pred_class = self.infer(self.resnet_interpreter, input_data)
            print(f"ResNet Confidence: {confidence}")
        
        return confidence, pred_class

# Preprocess the CIFAR-10 test image for inference
def preprocess_input(image):
    image = np.expand_dims(image, axis=0).astype(np.uint8)  # Convert to UINT8
    return image

dynamic_model = DynamicModelSwitching("mobilenet_int8.tflite", "resnet_int8.tflite")

# Test on a few CIFAR-10 images
for i in range(5):
    input_image = preprocess_input(x_test[i])
    print(f"Running inference on sample {i + 1}")
    confidence, predicted_class = dynamic_model.decide_model(input_image)
    print(f"Final Confidence: {confidence}, Predicted Class: {predicted_class}")

import time

def resource_cost_function(inference_time, cpu_util, memory_util):
    """Compute resource cost as a weighted sum of time, CPU, and memory usage"""
    return inference_time + 0.5 * cpu_util + 0.3 * memory_util

def measure_inference_time(interpreter, input_data):
    """Measure inference time of the given model"""
    start_time = time.time()
    interpreter.invoke()
    end_time = time.time()
    return end_time - start_time

# Example of using the cost function
input_image = preprocess_input(x_test[0])
cpu_util, memory_util = get_resource_utilization()
inference_time = measure_inference_time(dynamic_model.mobilenet_interpreter, input_image)
cost = resource_cost_function(inference_time, cpu_util, memory_util)
print(f"Resource cost: {cost}")
