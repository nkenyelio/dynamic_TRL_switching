import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.applications import ResNet50, MobileNetV2
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
import psutil
import tflite_runtime.interpreter as tflite
import GPUtil
import numpy as np

# Load CIFAR-10 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize data
num_classes = 10

# Prepare the CIFAR-10 dataset for training
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

# Define function to create models
def create_model(base_model):
    base_model.trainable = False  # Freeze the base model
    x = Flatten()(base_model.output)
    x = Dense(256, activation='relu')(x)
    output = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=base_model.input, outputs=output)

# Load ResNet and MobileNet
resnet_base = ResNet50(include_top=False, input_shape=(32, 32, 3), weights='imagenet')
mobilenet_base = MobileNetV2(include_top=False, input_shape=(32, 32, 3), weights='imagenet')

resnet_model = create_model(resnet_base)
mobilenet_model = create_model(mobilenet_base)

# Compile models
resnet_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
mobilenet_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train models
resnet_model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
mobilenet_model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# Save models for later use
resnet_model.save('resnet_cifar10.h5')
mobilenet_model.save('mobilenet_cifar10.h5')

# Convert ResNet model to TFLite with INT8 quantization
def convert_to_tflite_int8(model, filename):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative_dataset_gen():
        for i in range(100):
            yield [x_train[i:i+1]]

    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()
    with open(filename, 'wb') as f:
        f.write(tflite_model)

# Convert both models to TFLite INT8
convert_to_tflite_int8(resnet_model, 'resnet_cifar10_int8.tflite')
convert_to_tflite_int8(mobilenet_model, 'mobilenet_cifar10_int8.tflite')



# Load models
resnet_tflite_path = 'resnet_cifar10_int8.tflite'
mobilenet_tflite_path = 'mobilenet_cifar10_int8.tflite'

def load_model(model_path):
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def run_inference(interpreter, input_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    return interpreter.get_tensor(output_details[0]['index'])

# Step 2: System Resource Monitoring Functions
def get_cpu_usage():
    """Get current system CPU usage."""
    return psutil.cpu_percent(interval=1)

def get_gpu_usage():
    """Get current GPU memory usage."""
    gpus = GPUtil.getGPUs()
    if gpus:
        return max([gpu.memoryUtil for gpu in gpus]) * 100  # Get memory utilization in percentage
    return 40  # Return None if no GPU is available

def get_memory_usage():
    """Get current system memory usage."""
    return psutil.virtual_memory()


# Function to get system resources
def get_system_resources():
    cpu_usage = get_cpu_usage()
    memory_info = get_memory_usage()
    gpu_usage = get_gpu_usage()  # Use tegrastats for GPU monitoring on Jetson Nano
    return cpu_usage, memory_info.percent, gpu_usage


# Dynamic model switching based on CPU load
def choose_model(cpu_usage, gpu_usage, threshold=60):
    if cpu_usage > threshold or gpu_usage > threshold:
        print("Switching to MobileNet (lighter model) due to high resource usage")
        return mobilenet_tflite_path
    else:
        print("Using ResNet (heavier model)")
        return resnet_tflite_path

# Preprocess input for inference
def preprocess_input(image):
    image = np.expand_dims(image, axis=0)
    return np.array(image, dtype=np.int8)

# Simulate dynamic inference
while True:
    cpu_usage = get_cpu_usage()

    # Choose model based on CPU usage
    selected_model = choose_model(cpu_usage)
    interpreter = load_model(selected_model)

    # Dummy input (replace with real input in practice)
    input_image = x_test[0]
    input_data = preprocess_input(input_image)

    # Run inference
    output = run_inference(interpreter, input_data)

    print(f"Inference result: {np.argmax(output)}")

    # Sleep for a short period before next inference
    import time
    time.sleep(1)
