import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Resize images to (32, 32) and replicate to 3 channels
x_train = np.stack([x_train] * 3, axis=-1)  # Convert to 3 channels
x_test = np.stack([x_test] * 3, axis=-1)
x_train = tf.image.resize(x_train, [32, 32])  # Resize to 32x32
x_test = tf.image.resize(x_test, [32, 32])

# Normalize the pixel values
x_train = x_train / 255.0
x_test = x_test / 255.0

# One-hot encode the labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Build the ResNet50 model (without top layers)
base_model = ResNet50(include_top=False, weights=None, input_shape=(32, 32, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Add Global Average Pooling
x = Dense(128, activation='relu')(x)  # Add a dense layer
output = Dense(10, activation='softmax')(x)  # Output layer for 10 classes

model = Model(inputs=base_model.input, outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))

# Save the trained model
model.save('resnet50_mnist_model.h5')

# Convert the Keras model to TensorFlow Lite model with INT8 quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Enable quantization
tflite_model = converter.convert()

# Save the quantized model
with open('resnet50_mnist_int8_quantized.tflite', 'wb') as f:
    f.write(tflite_model)

print("Quantized INT8 model saved as resnet50_mnist_int8_quantized.tflite")

# Load the quantized model
interpreter = tf.lite.Interpreter(model_path="resnet50_mnist_int8_quantized.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Prepare input data (example using a single image)
input_data = np.expand_dims(x_test[0], axis=0).astype(np.float32)  # Prepare the input image
interpreter.set_tensor(input_details[0]['index'], input_data)

# Run inference
interpreter.invoke()

# Get prediction
output_data = interpreter.get_tensor(output_details[0]['index'])
print("Predicted label:", np.argmax(output_data))
