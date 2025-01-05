import tensorflow as tf
import psutil
import GPUtil
import time
import numpy as np
from tensorflow.keras import layers, models
import tensorflow.lite as tflite
from tensorflow.keras.applications import ResNet50, MobileNetV2
from tensorflow.keras.datasets import cifar10
from tensorflow_model_optimization.quantization.keras import quantize_model
import tensorflow_model_optimization as tfmot

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize data





def get_resource_utilization():
    """Return CPU and memory utilization percentages"""
    cpu_utilization = psutil.cpu_percent(interval=1)
    memory_utilization = psutil.virtual_memory().percent
    return cpu_utilization, memory_utilization

def get_gpu_usage():
    """Get current GPU memory usage."""
    gpus = GPUtil.getGPUs()
    if gpus:
        return max([gpu.memoryUtil for gpu in gpus]) * 100  # Get memory utilization in percentage
    return 60  # Return None if no GPU is available
class DynamicModelSwitching:
    def __init__(self, mobilenet_model_path, resnet_model_path, confidence_threshold=0.7, resource_limit=80):
        # Load the TFLite models
        self.mobilenet_interpreter = tflite.Interpreter(model_path=mobilenet_model_path)
        self.resnet_interpreter = tflite.Interpreter(model_path=resnet_model_path)
        self.confidence_threshold = confidence_threshold
        self.resource_limit = resource_limit  # Max CPU/memory allowed

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


def resource_cost_function(inference_time, cpu_util, memory_util):
    """Compute resource cost as a weighted sum of time, CPU, and memory usage"""
    return inference_time + 0.5 * cpu_util + 0.3 * memory_util


def measure_inference_time(interpreter, input_data):
    """Measure inference time of the given model"""
    start_time = time.time()
    interpreter.invoke()
    end_time = time.time()
    return end_time - start_time


dynamic_model = DynamicModelSwitching("mobilenet_qat_int8_qat.tflite", "resnet_qat_int8_qat.tflite")
# Example of using the cost function
#input_image = np.expand_dims(x_test[0], axis=0).astype(np.float32)
#cpu_util, memory_util = get_resource_utilization()
#inference_time = measure_inference_time(dynamic_model.mobilenet_interpreter, input_image)
#cost = resource_cost_function(inference_time, cpu_util, memory_util)
#print(f"Resource cost: {cost}")

for epoch in range(5):
        cost = 0.0
        #if weights is None:
        weights = {'accuracy': 0.4, 'confidence': 0.3, 'cpu': 0.1, 'gpu': 0.1, 'memory': 0.1,
                   'complexity': 0.2}  # Default weights
        task_complexity = 1
        for i in range(5):
            input_image = np.expand_dims(x_test[i], axis=0).astype(np.float32)
            print(f"Running inference on sample {i + 1}")
            accuracy = 0.6
            cpu_util, memory_util = get_resource_utilization()
            gpu_usage = get_gpu_usage()
            print(f"Resource CPU: {cpu_util}, Resource Memory: {memory_util}, Resource GPU: {gpu_usage}")
            inference_time = measure_inference_time(dynamic_model.mobilenet_interpreter, input_image)
            print(f"Inference_time: {inference_time}")
            #cost = resource_cost_function(inference_time, cpu_util, memory_util)
            confidence, predicted_class = dynamic_model.decide_model(input_image)
            print(f"Final Confidence: {confidence}, Predicted Class: {predicted_class}")
            cost = compute_resource_cost(accuracy, confidence, cpu_util, gpu_usage, memory_util, task_complexity,
                                             weights)
            print(f"Resource cost: {cost}")
        print(f"Epoch [{epoch + 1}/{5}], Cost: {cost}")
print("Dynamic switching complete")

