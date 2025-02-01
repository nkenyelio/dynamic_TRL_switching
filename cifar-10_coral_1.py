import numpy as np
import os
from PIL import Image
from pycoral.utils.edgetpu import make_interpreter

from pycoral.adapters import common
import time

# Path to the CIFAR-10 data directory
data_dir = "data/cifar-10-batches-py/"
# Function to preprocess the image
def preprocess_image(image):
    image = Image.fromarray(image).resize((input_details[0]['shape'][2], input_details[0]['shape'][1]))
    image = np.array(image, dtype=np.float32)
    image = image / 255.0  # Normalize to [0, 1]
    return image
def resource_consumption(latency, memory, flops, tpu_usage, accuracy):
    weights = {"latency": 0.4, "memory": 0.2, "flops": 0.2, "tpu_usage": 0.1, "accuracy": 0.1}
    return (
        weights["latency"] * latency +
        weights["memory"] * memory +
        weights["flops"] * flops +
        weights["tpu_usage"] * tpu_usage -
        weights["accuracy"] * accuracy  # Maximize accuracy
    )
# Function to load CIFAR-10 images
def load_cifar10_images(data_dir, num_images=50):
    images = []
    labels = []
    
    # CIFAR-10 binary files are named like 'data_batch_1', 'data_batch_2', etc.
    for batch_file in os.listdir(data_dir):
        if batch_file.startswith('data_batch_'):
            with open(os.path.join(data_dir, batch_file), 'rb') as f:
                data = np.frombuffer(f.read(), dtype=np.uint8)
                data = data.reshape(-1, 3073)  # Each image is 32x32x3 + 1 label
                batch_images = data[:, 1:].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
                batch_labels = data[:, 0]
                
                images.extend(batch_images)
                labels.extend(batch_labels)
                
                if len(images) >= num_images:
                    break
    
    # Convert to numpy arrays and limit to the first `num_images`
    images = np.array(images[:num_images])
    labels = np.array(labels[:num_images])
    
    return images, labels

# Load the first 50 images and their labels
images, labels = load_cifar10_images(data_dir, num_images=50)

# Path to the Edge TPU model
model_paths = {
    "model-1": "cifar10_model_edgetpu.tflite",
    "nasanet": "nasnet_edge_cifar10_edgetpu.tflite",
    "vgg": "vgg_edge_cifar10_edgetpu.tflite",
    "densenet": "densenet_edge_cifar10_edgetpu.tflite",
    "resnet": "resnet_edge_cifar10_edgetpu.tflite",
    "mobilenet": "mobilenet_edge_cifar10_edgetpu.tflite",
}
best_model = None
best_score = float("inf")
application_complexity = 0.5

for model_name, model_path in model_paths.items():
    # Initialize the interpreter
    interpreter = make_interpreter(model_path)
    interpreter.allocate_tensors()

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    start_time = time.time()
    results = []
    for image in images:
        # Preprocess the image
        preprocessed_image = preprocess_image(image)

        # Set the input tensor
        common.set_input(interpreter, preprocessed_image)

        # Run inference
        interpreter.invoke()

        # Get the output
        output = interpreter.get_tensor(output_details[0]['index'])
        results.append(output)
    latency = time.time() - start_time
    # Calculate accuracy
    correct_predictions = 0
    for result, label in zip(results, labels):
        predicted_label = np.argmax(result)
        if predicted_label == label:
            correct_predictions += 1
    #accuracy = correct_predictions / len(results)
    accuracy = np.random.uniform(0.8, 0.99)  # Simulated accuracy
    memory = np.random.uniform(50, 150)  # Simulated memory in MB
    flops = np.random.uniform(1e6, 1e8)  # Simulated FLOPs
    tpu_usage = np.random.uniform(40, 80)  # Simulated TPU usage in percentage
    score = resource_consumption(latency, memory, flops, tpu_usage, accuracy)
    adjusted_score = score * (1 + application_complexity)
    print(f"Model: {model_name}, Adjusted Score: {adjusted_score:.4f}, Latency: {latency:.4f}s")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    if adjusted_score < best_score:
        best_score = adjusted_score
        best_model = model_name
    output_file = "coco/cifar_inference_results_" + str(len(images)) + model_name + ".txt"
    with open(output_file, "w") as f:
        for i, (result, label) in enumerate(zip(results, labels)):
            predicted_label = np.argmax(result)
            f.write(f"Image {i+1}: Predicted={predicted_label}, Actual={label}\n")
print(f"Best Model is : {best_model}\n")
