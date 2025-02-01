import json
import pickle
import numpy as np
import os
from PIL import Image

from pycoral.utils.edgetpu import make_interpreter

import numpy as np
import time

# Path to your COCO annotations and images
#annotations_path = 'coco/annotations_trainval2017/annotations/instances_train2017.json'
images_dir = 'coco/val2017/'
image_files = [os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
images = []
for image_path in image_files[:50]:
    image = Image.open(image_path).convert('RGB')   # Convert to RGB format
    images.append(image)
print(f"Loaded {len(images)} images.")

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
def preprocess_image(image, input_shape):
    image = image.resize(input_shape)
    image = np.array(image, dtype=np.float32)
    image = image / 255.0  # Normalize if required
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image
model_paths = {
    "densenet": "densenet_coco_edgetpu.tflite",
    "efficientnet": "efficientnet-edgetpu-L_quant_edgetpu.tflite",
    "inceptionresnet": "inceptionresnet_coco_edgetpu.tflite",
    "vgg": "vgg_mnist_quantized_edgetpu.tflite",
    "inception_v3": "inception_v3_299_quant_edgetpu.tflite",
    "mobilenet": "mobilenet_coco_edgetpu.tflite",
     "mobilenet_v2": " mobilenet_v2_1.0_224_quant_edgetpu.tflite",
    "nasnetmobile": "nasnetmobile_coco_edgetpu.tflite",
    "resnet": "resnet_coco_edgetpu.tflite",
    "ssd_mobilenet_v1": "ssd_mobilenet_v1_coco_quant_postprocess_edgetpu.tflite",
    "ssd_mobilenet_v2": "ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite",
}
best_model = None
best_score = float("inf")
application_complexity = 0.5

for model_name, model_path in model_paths.items():
    interpreter = make_interpreter(model_path)
    interpreter.allocate_tensors()
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = tuple(input_details[0]['shape'][1:3])
    start_time = time.time()
    results = []
    for image in images:
        # Preprocess the image
        preprocessed_image = preprocess_image(image, input_shape)

        # Set the input tensor
        interpreter.set_tensor(input_details[0]['index'], preprocessed_image)

        # Run inference
        interpreter.invoke()

        # Get the output tensor
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # Store the results
        results.append(output_data)
    latency = time.time() - start_time
    accuracy = np.random.uniform(0.8, 0.99)  # Simulated accuracy
    memory = np.random.uniform(50, 150)  # Simulated memory in MB
    flops = np.random.uniform(1e6, 1e8)  # Simulated FLOPs
    tpu_usage = np.random.uniform(40, 80)  # Simulated TPU usage in percentage
    score = resource_consumption(latency, memory, flops, tpu_usage, accuracy)
    adjusted_score = score * (1 + application_complexity)
    print(f"Model: {model_name}, Adjusted Score: {adjusted_score:.4f}, Latency: {latency:.4f}s")
    if adjusted_score < best_score:
        best_score = adjusted_score
        best_model = model_name
    for i, result in enumerate(results):
        predicted_class = np.argmax(result)
        print(f"Image {i+1} predicted class: {predicted_class}")
    #save the results to a file
    outputfile = "coco/results_" + model_name + ".pkl"
    with open(outputfile, "wb") as f:
        pickle.dump(results, f)
    print(f"Results saved to {outputfile}.")
print(f"Best Model is : {best_model}\n")
