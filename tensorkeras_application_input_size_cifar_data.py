import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np
import psutil
import GPUtil
import time

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

def build_lightweight_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10)
    ])
    return model
def build_complex_model():
    model = models.Sequential([
        layers.Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dense(10)
    ])
    return model


def resource_cost_function():
    memory_info = psutil.virtual_memory()
    cpu_usage = psutil.cpu_percent(interval=1)

    if memory_info.available < 500_000_000:  # 500MB memory threshold
        return 'lightweight'
    elif cpu_usage > 75:  # 75% CPU usage threshold
        return 'lightweight'
    else:
        return 'complex'

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
def inference_confidence(predictions, threshold=0.7):
    confidence_scores = tf.nn.softmax(predictions, axis=-1)
    max_confidence = np.max(confidence_scores)
    return max_confidence >= threshold

def run_inference_confidence(predictions):
    confidence_scores = tf.nn.softmax(predictions, axis=-1)
    max_confidence = np.max(confidence_scores)
    return max_confidence

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


def measure_inference_latency(model, input_data):
    start_time = time.time()
    predictions = model(input_data)
    latency = time.time() - start_time
    return predictions, latency
def dynamic_model_switching(input_data, lightweight_model, complex_model):
    model_choice = resource_cost_function()

    if model_choice == 'lightweight':
        print("Using Lightweight Model")
        predictions = lightweight_model.predict(input_data)
        if not inference_confidence(predictions):
            print("Switching to Complex Model due to low confidence")
            predictions = complex_model.predict(input_data)
    else:
        print("Using Complex Model")
        predictions = complex_model.predict(input_data)

    return predictions
# Compile models
lightweight_model = build_lightweight_model()
complex_model = build_complex_model()

lightweight_model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
complex_model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

# Train lightweight model
lightweight_model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test), batch_size=64)

# Train complex model
complex_model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test), batch_size=64)
for i in range(10):  # Test on 5 random samples
    sample = np.expand_dims(x_test[i], axis=0)
    #cost = 0.0
    #if weights is None:
    weights = {'accuracy': 0.4, 'confidence': 0.3, 'cpu': 0.1, 'gpu': 0.1, 'memory': 0.1,
                   'complexity': 0.2}  # Default weights
    task_complexity = 1
    accuracy = 0.6
    cpu_util, memory_util = get_resource_utilization()
    gpu_usage = get_gpu_usage()
    print(f"Resource CPU: {cpu_util}, Resource Memory: {memory_util}, Resource GPU: {gpu_usage}")
    #inference_time = measure_inference_time(dynamic_model.mobilenet_interpreter, input_image)
    #print(f"Inference_time: {inference_time}")
    #confidence = 0.47
    output = dynamic_model_switching(sample, lightweight_model, complex_model)
    predict_1, inference_time = measure_inference_latency(lightweight_model, sample)
    print(f"Inference time: {inference_time}")
    confidence = run_inference_confidence(output)
    cost = compute_resource_cost(accuracy, confidence, cpu_util, gpu_usage, memory_util, task_complexity,
                                             weights)
    print(f"Resource cost: {cost}")
    print(f"Predicted class: {np.argmax(output)}")
