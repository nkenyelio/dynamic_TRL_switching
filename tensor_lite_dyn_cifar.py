import tensorflow as tf
import psutil
import time
import numpy as np
import tensorflow.lite as tflite
from tensorflow.keras.applications import ResNet50, MobileNetV2
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow_model_optimization.quantization.keras import quantize_model
import tensorflow_model_optimization as tfmot

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
#x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize data


#Define a simple model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile and train a simple model

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=64, epochs = 10, validation_data=(x_test, y_test))

#convert to TFLIte

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("cifar10_model_trrt.tflite", "wb") as f:
    f.write(tflite_model)
# Create and Quantize MobileNetV2 with QAT
def create_mobilenet_qat():
    base_model = MobileNetV2(weights=None, input_shape=(32, 32, 3), classes=10)
    # Quantize the model using QAT
    qat_model = tfmot.quantization.keras.quantize_model(base_model)
    qat_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return qat_model

# Create and Quantize ResNet50 with QAT
def create_resnet_qat():
    base_model = ResNet50(weights=None, input_shape=(32, 32, 3), classes=10)
    qat_model = tfmot.quantization.keras.quantize_model(base_model)
    qat_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return qat_model

# Train QAT models
mobilenet_qat = create_mobilenet_qat()
resnet_qat = create_resnet_qat()

mobilenet_qat.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
resnet_qat.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

def convert_to_tflite_qat(model, model_name):
    """Convert QAT Keras model to TensorFlow Lite INT8 model"""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    # Save the TFLite model
    with open(f"{model_name}_int8_qat.tflite", "wb") as f:
        f.write(tflite_model)
    print(f"Model saved: {model_name}_int8_qat.tflite")

convert_to_tflite_qat(mobilenet_qat, "mobilenet_qat")
convert_to_tflite_qat(resnet_qat, "resnet_qat")


def get_resource_utilization():
    """Return CPU and memory utilization percentages"""
    cpu_utilization = psutil.cpu_percent(interval=1)
    memory_utilization = psutil.virtual_memory().percent
    return cpu_utilization, memory_utilization



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
input_image = np.expand_dims(x_test[0], axis=0).astype(np.uint8)
cpu_util, memory_util = get_resource_utilization()
inference_time = measure_inference_time(dynamic_model.mobilenet_interpreter, input_image)
cost = resource_cost_function(inference_time, cpu_util, memory_util)
print(f"Resource cost: {cost}")

dynamic_model = DynamicModelSwitching("mobilenet_qat_int8.tflite", "resnet_qat_int8.tflite")

for i in range(5):
    input_image = np.expand_dims(x_test[i], axis=0).astype(np.uint8)
    print(f"Running inference on sample {i + 1}")
    confidence, predicted_class = dynamic_model.decide_model(input_image)
    print(f"Final Confidence: {confidence}, Predicted Class: {predicted_class}")
