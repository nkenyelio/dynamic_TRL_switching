import numpy as np
from PIL import Image
import os
from tflite_runtime.interpreter import Interpreter, load_delegate
import time

data = np.load('archive/mnist_compressed.npz')
x_test = data['test_images']
y_test = data['test_labels']
# Resource consumption functioninput_image
def resource_consumption(latency, memory, flops, tpu_usage, accuracy):
    weights = {"latency": 0.4, "memory": 0.2, "flops": 0.2, "tpu_usage": 0.1, "accuracy": 0.1}
    return (
        weights["latency"] * latency +
        weights["memory"] * memory +
        weights["flops"] * flops +
        weights["tpu_usage"] * tpu_usage -
        weights["accuracy"] * accuracy  # Maximize accuracy
    )
model_paths = {
    "mobilenet": "mnist_model_edgetpu.tflite",
    "resnet": "resnet_mnist_quantized_edgetpu.tflite",
    #"densenet": "densenet_mnist_quantized_edgetpu.tflite",
    "vgg": "vgg_mnist_quantized_edgetpu.tflite",
    #"densenet": "densenet_edgetpu.tflite",
    #"yolo": "yolo_edgetpu.tflite",
}
best_model = None
best_score = float("inf")
application_complexity = 0.5

for model_name, model_path in model_paths.items():
    interpreter = Interpreter(model_path, experimental_delegates=[load_delegate('libedgetpu.so.1')])
    interpreter.allocate_tensors()
    # Get input and outputs details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    x_test = x_test.reshape((-1, 28, 28, 1)).astype("float32") / 255.0
    images = x_test[:100]
    labels = y_test[:100]
    #Perform inference
    predictions = []
    start_time = time.time()
    for i in range(len(images)):
        #set input tensor
        interpreter.set_tensor(input_details[0]['index'], [images[i]])

        #Run inference

        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        predictions.append(np.argmax(output))
    latency = time.time() - start_time
    accuracy = np.mean(np.array(predictions) == labels) # real accuracy accuracy#
    memory = np.random.uniform(50, 150)  # Simulated memory in MB
    flops = np.random.uniform(1e6, 1e8)  # Simulated FLOPs
    tpu_usage = np.random.uniform(40, 80)  # Simulated TPU usage in percentage
    score = resource_consumption(latency, memory, flops, tpu_usage, accuracy)
    adjusted_score = score * (1 + application_complexity)
    print(f"Model: {model_name}, Adjusted Score: {adjusted_score:.4f}, Latency: {latency:.4f}s")
    if adjusted_score < best_score:
        best_score = adjusted_score
        best_model = model_name
print(f"Best Model is : {best_model}\n")
