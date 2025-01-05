import tensorflow as tf
import tflite_runtime.interpreter as tflite
import time
import psutil
import GPUtil
#import torch
#import torch.nn as nn
#import torch.optim as optim
#from torchvision import models



##import tensorflow as tf

# Load pre-trained ResNet and MobileNet models
resnet_model = tf.keras.applications.ResNet50(weights='imagenet')
mobilenet_model = tf.keras.applications.MobileNetV2(weights='imagenet')

# Convert ResNet to INT8
resnet_converter = tf.lite.TFLiteConverter.from_keras_model(resnet_model)
resnet_converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Representative dataset generator for ResNet
def representative_data_gen_resnet():
    for input_value in dataset_resnet:  # Your dataset for ResNet calibration
        yield [input_value]

resnet_converter.representative_dataset = representative_data_gen_resnet
resnet_converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
resnet_converter.inference_input_type = tf.int8
resnet_converter.inference_output_type = tf.int8
resnet_tflite_model = resnet_converter.convert()

# Save INT8 ResNet model
with open('resnet_int8.tflite', 'wb') as f:
    f.write(resnet_tflite_model)

# Convert MobileNet to INT8
mobilenet_converter = tf.lite.TFLiteConverter.from_keras_model(mobilenet_model)
mobilenet_converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Representative dataset generator for MobileNet
def representative_data_gen_mobilenet():
    for input_value in dataset_mobilenet:  # Your dataset for MobileNet calibration
        yield [input_value]

mobilenet_converter.representative_dataset = representative_data_gen_mobilenet
mobilenet_converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
mobilenet_converter.inference_input_type = tf.int8
mobilenet_converter.inference_output_type = tf.int8
mobilenet_tflite_model = mobilenet_converter.convert()

# Save INT8 MobileNet model
with open('mobilenet_int8.tflite', 'wb') as f:
    f.write(mobilenet_tflite_model)

# Step 2: System Resource Monitoring Functions
def get_cpu_usage():
    """Get current system CPU usage."""
    return psutil.cpu_percent(interval=1)

def get_gpu_usage():
    """Get current GPU memory usage."""
    gpus = GPUtil.getGPUs()
    if gpus:
        return max([gpu.memoryUtil for gpu in gpus]) * 100  # Get memory utilization in percentage
    return None  # Return None if no GPU is available

def get_memory_usage():
    """Get current system memory usage."""
    return psutil.virtual_memory()


# Function to get system resources
def get_system_resources():
    cpu_usage = get_cpu_usage()
    memory_info = get_memory_usage()
    gpu_usage = get_gpu_usage()  # Use tegrastats for GPU monitoring on Jetson Nano
    return cpu_usage, memory_info.percent, gpu_usage



# Load models dynamically
def load_model(model_path):
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

# Paths to INT8 models
resnet_int8_path = 'resnet_int8.tflite'
mobilenet_int8_path = 'mobilenet_int8.tflite'

# Select model based on resource usage
def choose_model(cpu_usage, gpu_usage, threshold=60):
    if cpu_usage > threshold or gpu_usage > threshold:
        print("Switching to MobileNet (lighter model) due to high resource usage")
        return mobilenet_int8_path
    else:
        print("Using ResNet (heavier model)")
        return resnet_int8_path

# Run inference
def run_inference(model_path, input_data):
    interpreter = load_model(model_path)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Prepare input
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    # Perform inference
    interpreter.invoke()
    
    # Get output
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

# Main loop for dynamic model switching
while True:
    # Monitor system resources
    cpu_usage, memory_usage, gpu_usage = get_system_resources()

    # Choose model based on resource availability
    selected_model_path = choose_model(cpu_usage, gpu_usage)

    # Dummy input data (replace with actual data)
    input_data = ...

    # Run inference using the selected model
    output = run_inference(selected_model_path, input_data)

    # Handle the output (e.g., print or process it)
    print(f"Inference Output: {output}")

    # Pause for a while before the next inference
    time.sleep(1)
