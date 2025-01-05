import tensorflow as tf
from tensorflow.keras.applications import ResNet50, MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import numpy as np
import tensorflow_model_optimization as tfmot

# Load and preprocess CIFAR-10 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize the input data
x_train, x_test = x_train / 255.0, x_test / 255.0

# One-hot encode the labels
y_train, y_test = to_categorical(y_train, 10), to_categorical(y_test, 10)

# Resize CIFAR-10 to match ResNet50 input size (32, 32, 3 -> 224, 224, 3)
x_train_resized = tf.image.resize(x_train, [224, 224])
x_test_resized = tf.image.resize(x_test, [224, 224])

# Define MobileNetV2 Model for lightweight tasks
def create_mobilenet():
    base_model = MobileNetV2(include_top=False, weights=None, input_shape=(32, 32, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    output = Dense(10, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)
    return model

# Define ResNet50 Model for complex tasks
def create_resnet():
    base_model = ResNet50(include_top=False, weights=None, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    output = Dense(10, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)
    return model

# Create QAT-aware MobileNet and ResNet50 models
def apply_qat(model):
    # Apply quantization-aware training
    quantize_model = tfmot.quantization.keras.quantize_model
    qat_model = quantize_model(model)
    return qat_model

mobilenet_model = create_mobilenet()
resnet_model = create_resnet()

# Compile the models
mobilenet_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
resnet_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Apply QAT to both models
qat_mobilenet_model = apply_qat(mobilenet_model)
qat_resnet_model = apply_qat(resnet_model)

# Train the models
qat_mobilenet_model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))
qat_resnet_model.fit(x_train_resized, y_train, epochs=10, batch_size=64, validation_data=(x_test_resized, y_test))

# Convert to TensorFlow Lite with INT8 quantization
def convert_to_tflite(model, filename):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    with open(filename, 'wb') as f:
        f.write(tflite_model)

convert_to_tflite(qat_mobilenet_model, 'mobilenet_qat_int8.tflite')
convert_to_tflite(qat_resnet_model, 'resnet_qat_int8.tflite')

print("Models have been quantized and saved.")

# Dynamic Model Switching Logic Based on Resource Constraints
def dynamic_model_switching(input_data, complexity_level, available_memory, inference_confidence):
    """
    A function that switches between lightweight (MobileNet) and complex (ResNet) models based on:
    - Task complexity (e.g., low or high)
    - System resource constraints (e.g., available memory)
    - Inference confidence (how confident the model should be)
    """
    if complexity_level == 'low' and available_memory > 100:  # Example condition
        print("Using MobileNet for inference")
        interpreter = tf.lite.Interpreter(model_path="mobilenet_qat_int8.tflite")
    else:
        print("Using ResNet for inference")
        interpreter = tf.lite.Interpreter(model_path="resnet_qat_int8.tflite")
    
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
result = dynamic_model_switching(x_test[0], complexity_level='high', available_memory=150, inference_confidence=0.9)
print("Inference Result:", np.argmax(result))
