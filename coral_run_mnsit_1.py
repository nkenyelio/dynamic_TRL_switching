import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import (
    MobileNet, ResNet50, EfficientNetB0, VGG16, DenseNet121
)
from PIL import Image
from tflite_runtime.interpreter import Interpreter, load_delegate
import time

# Load and preprocess MNIST dataset
def load_and_preprocess_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    return (x_train, y_train), (x_test, y_test)

# Quantization and conversion function
def quantize_and_compile_model(saved_model_dir, output_tflite, compiled_model_path):
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    tflite_model = converter.convert()
    
    with open(output_tflite, "wb") as f:
        f.write(tflite_model)
    
    # Compile for Edge TPU
    import os
    os.system(f"edgetpu_compiler {output_tflite} --output_dir {compiled_model_path}")

# Resource consumption function
def resource_consumption(latency, memory, flops, tpu_usage, accuracy):
    weights = {"latency": 0.4, "memory": 0.2, "flops": 0.2, "tpu_usage": 0.1, "accuracy": 0.1}
    return (
        weights["latency"] * latency +
        weights["memory"] * memory +
        weights["flops"] * flops +
        weights["tpu_usage"] * tpu_usage -
        weights["accuracy"] * accuracy  # Maximize accuracy
    )

# Perform inference on a single image and log metrics
def infer_on_model(model_path, input_image):
    interpreter = Interpreter(model_path=model_path, experimental_delegates=[load_delegate('libedgetpu.so.1')])
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Set input
    interpreter.set_tensor(input_details[0]['index'], input_image)
    
    # Measure latency
    start_time = time.time()
    interpreter.invoke()
    latency = time.time() - start_time
    
    # Extract output and simulate accuracy and resource usage
    output = interpreter.get_tensor(output_details[0]['index'])
    accuracy = np.random.uniform(0.8, 0.95)  # Simulated accuracy
    memory = np.random.uniform(50, 150)  # Simulated memory in MB
    flops = np.random.uniform(1e8, 1e10)  # Simulated FLOPs
    tpu_usage = np.random.uniform(40, 80)  # Simulated TPU usage in percentage
    
    return latency, memory, flops, tpu_usage, accuracy

# Preprocess image for inference
def preprocess_image(image, target_size=(224, 224)):
    image = Image.fromarray(image).convert("RGB")
    image = image.resize(target_size)
    image = np.array(image, dtype=np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Dynamic model switching
def dynamic_model_switch(test_image, models_paths, application_complexity=0.5):
    input_image = preprocess_image(test_image)
    best_model = None
    best_score = float("inf")
    
    for model_name, model_path in models_paths.items():
        latency, memory, flops, tpu_usage, accuracy = infer_on_model(model_path, input_image)
        score = resource_consumption(latency, memory, flops, tpu_usage, accuracy)
        adjusted_score = score * (1 + application_complexity)
        print(f"Model: {model_name}, Adjusted Score: {adjusted_score:.4f}, Latency: {latency:.4f}s")
        
        if adjusted_score < best_score:
            best_score = adjusted_score
            best_model = model_name
    
    return best_model

# Main execution
if __name__ == "__main__":
    # Load MNIST data
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_mnist()
    
    # Define model paths (quantized and compiled for Edge TPU)
    models_paths = {
        "mobilenet": "mobilenet_edgetpu.tflite",
        "resnet": "resnet_edgetpu.tflite",
        "efficientnet": "efficientnet_edgetpu.tflite",
        "vgg": "vgg_edgetpu.tflite",
        "densenet": "densenet_edgetpu.tflite",
        "yolo": "yolo_edgetpu.tflite",
    }
    
    # Test dynamic switching
    test_image = x_test[0]
    best_model = dynamic_model_switch(test_image, models_paths)
    print(f"Best Model for the given input: {best_model}")
