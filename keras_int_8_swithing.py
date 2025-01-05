import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import psutil
import numpy as np
import tensorflow.lite as tflite
import time

# Load and preprocess CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize images

y_train, y_test = to_categorical(y_train), to_categorical(y_test)



def create_resnet_qat():
    base_model = tf.keras.applications.ResNet50(weights=None, input_shape=(32, 32, 1), classes=10)
    model = tf.keras.Sequential([
      tf.keras.layers.InputLayer(input_shape=(32, 32)),
      tf.keras.layers.Reshape(target_shape=(32, 32, 1)),
      tf.keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation='relu'),
      tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(10)])

# Train the digit classification model
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#    model.fit(
 # x_train,
 # x_test,
 # epochs=1,
#  validation_split=0.1,)

    #model = tf.keras.Model(base_model.input, x)
    #model.summary()
    #model = tf.keras.models.Sequential([base_model, tf.keras.layers.Softmax()])
    
    # Apply Quantization-Aware Training (QAT)
    qat_model = tfmot.quantization.keras.quantize_model(model)
    qat_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return qat_model

def create_mobilenet_qat():
    base_model = tf.keras.applications.MobileNetV2(weights=None, input_shape=(32, 32, 1), classes=10)
    #base_model.trainable = False
    #model = tf.keras.models.Sequential([base_model, tf.keras.layers.Softmax()])

    model = tf.keras.Sequential([
      tf.keras.layers.InputLayer(input_shape=(32, 32)),
      tf.keras.layers.Reshape(target_shape=(32, 32, 1)),
      tf.keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation='relu'),
      tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(10)])

# Train the digit classification model
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    # Apply Quantization-Aware Training (QAT)
    qat_model = tfmot.quantization.keras.quantize_model(model)
    qat_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return qat_model

# Create models
resnet_qat = create_resnet_qat()
mobilenet_qat = create_mobilenet_qat()

# Train both models
resnet_qat.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
mobilenet_qat.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

def convert_qat_to_tflite(qat_model, model_name):
    converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    # Save the converted model
    with open(f"{model_name}_int8_qat.tflite", "wb") as f:
        f.write(tflite_model)
    print(f"Model saved: {model_name}_int8_qat.tflite")

# Convert both models
convert_qat_to_tflite(resnet_qat, "resnet_qat")
convert_qat_to_tflite(mobilenet_qat, "mobilenet_qat")



class DynamicModelSwitching:
    def __init__(self, mobilenet_model_path, resnet_model_path, confidence_threshold=0.7, resource_limit=80):
        # Load TensorFlow Lite interpreters
        self.mobilenet_interpreter = tflite.Interpreter(model_path=mobilenet_model_path)
        self.resnet_interpreter = tflite.Interpreter(model_path=resnet_model_path)
        self.mobilenet_interpreter.allocate_tensors()
        self.resnet_interpreter.allocate_tensors()
        
        self.confidence_threshold = confidence_threshold
        self.resource_limit = resource_limit

    def infer(self, interpreter, input_data):
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Prepare input data
        interpreter.set_tensor(input_details[0]['index'], input_data)
        
        # Run inference
        interpreter.invoke()
        
        # Get prediction and confidence
        output = interpreter.get_tensor(output_details[0]['index'])
        confidence = np.max(output)
        predicted_class = np.argmax(output)
        return confidence, predicted_class

    def get_resource_utilization(self):
        cpu_utilization = psutil.cpu_percent(interval=1)
        memory_utilization = psutil.virtual_memory().percent
        return cpu_utilization, memory_utilization

    def switch_model(self, input_data):
        cpu_util, memory_util = self.get_resource_utilization()
        print(f"CPU: {cpu_util}%, Memory: {memory_util}%")
        
        # Infer with MobileNet first
        confidence, predicted_class = self.infer(self.mobilenet_interpreter, input_data)
        print(f"MobileNet Confidence: {confidence}")

        # If confidence is low and resources allow, switch to ResNet
        if confidence < self.confidence_threshold and cpu_util < self.resource_limit:
            print("Switching to ResNet for better accuracy...")
            confidence, predicted_class = self.infer(self.resnet_interpreter, input_data)
            print(f"ResNet Confidence: {confidence}")
        
        return confidence, predicted_class


def resource_cost_function(inference_time, cpu_util, memory_util):
    """Compute resource cost as a weighted sum of time, CPU, and memory usage."""
    return inference_time + 0.5 * cpu_util + 0.3 * memory_util

def measure_inference_time(interpreter, input_data):
    """Measure inference time for the model."""
    start_time = time.time()
    interpreter.invoke()
    end_time = time.time()
    return end_time - start_time

dynamic_model = DynamicModelSwitching("mobilenet_qat_int8.tflite", "resnet_qat_int8.tflite")

# Test on CIFAR-10
for i in range(5):
    input_image = np.expand_dims(x_test[i], axis=0).astype(np.uint8)
    print(f"Running inference on sample {i + 1}")
    confidence, predicted_class = dynamic_model.switch_model(input_image)
    print(f"Final Confidence: {confidence}, Predicted Class: {predicted_class}")
