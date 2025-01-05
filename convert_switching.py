import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import tensorflow_model_optimization as tfmot
import numpy as np

# Load and preprocess MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Resize MNIST to 32x32 for MobileNetV2 input
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)
x_train = tf.image.resize(x_train, [32, 32])
x_test = tf.image.resize(x_test, [32, 32])

# Normalize the input data
x_train, x_test = x_train / 255.0, x_test / 255.0

# One-hot encode the labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Define a simple CNN model for less complex tasks
def create_simple_cnn():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model

# Define MobileNetV2 model for more complex tasks
def create_mobilenetv2_model():
    base_model = MobileNetV2(input_shape=(32, 32, 1), include_top=False, weights=None)
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    output = layers.Dense(10, activation='softmax')(x)
    model = models.Model(inputs=base_model.input, outputs=output)
    return model

# Apply Quantization-Aware Training (QAT) to a model
def apply_qat(model):
    quantize_model = tfmot.quantization.keras.quantize_model
    qat_model = quantize_model(model)
    return qat_model

# Create models
simple_cnn_model = create_simple_cnn()
mobilenetv2_model = create_mobilenetv2_model()

# Apply QAT to both models
qat_simple_cnn_model = apply_qat(simple_cnn_model)
qat_mobilenetv2_model = apply_qat(mobilenetv2_model)

# Compile both models
qat_simple_cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
qat_mobilenetv2_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train both models
qat_simple_cnn_model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
qat_mobilenetv2_model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Convert models to TensorFlow Lite with INT8 quantization
def convert_to_tflite(model, filename):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    with open(filename, 'wb') as f:
        f.write(tflite_model)

# Convert both models
convert_to_tflite(qat_simple_cnn_model, 'simple_cnn_qat_int8_mnist.tflite')
convert_to_tflite(qat_mobilenetv2_model, 'mobilenetv2_qat_int8_mnist.tflite')

print("Models converted and saved.")

# Dynamic Model Switching Logic Based on Resource Awareness
def dynamic_model_switching(input_data, task_complexity, available_memory, inference_confidence):
    """
    A function that switches between lightweight (simple CNN) and complex (MobileNetV2) models based on:
    - Task complexity (e.g., low or high)
    - System resource constraints (e.g., available memory)
    - Inference confidence (how confident the model should be)
    """
    if task_complexity == 'low' and available_memory > 100:  # Example condition
        print("Using Simple CNN for inference")
        interpreter = tf.lite.Interpreter(model_path="simple_cnn_qat_int8_mnist.tflite")
    else:
        print("Using MobileNetV2 for inference")
        interpreter = tf.lite.Interpreter(model_path="mobilenetv2_qat_int8_mnist.tflite")
    
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Set the input tensor
    input_data = np.expand_dims(input_data, axis=0).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get the result
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

# Example of switching based on resource awareness
result = dynamic_model_switching(x_test[0], task_complexity='high', available_memory=150, inference_confidence=0.9)
print("Inference Result:", np.argmax(result))
