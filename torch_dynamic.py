import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.quantization as quantization
import psutil
import GPUtil
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models
from torch.quantization import default_qconfig, default_per_channel_qconfig

# Define device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load CIFAR-10 dataset
transform = transforms.Compose([
    transforms.Resize(224),  # Resize for MobileNet/ResNet
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Load Pretrained ResNet and MobileNet models
resnet_model = models.resnet18(pretrained=True)
mobilenet_model = models.mobilenet_v2(pretrained=True)

# Modify the last layer for CIFAR-10 (10 classes)
resnet_model.fc = nn.Linear(resnet_model.fc.in_features, 10)
mobilenet_model.classifier[1] = nn.Linear(mobilenet_model.classifier[1].in_features, 10)

resnet_model = resnet_model.to(device)
mobilenet_model = mobilenet_model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer_resnet = optim.Adam(resnet_model.parameters(), lr=0.001)
optimizer_mobilenet = optim.Adam(mobilenet_model.parameters(), lr=0.001)

# Training function
def train_model(model, optimizer, train_loader, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}')

# Train both models
train_model(resnet_model, optimizer_resnet, train_loader)
train_model(mobilenet_model, optimizer_mobilenet, train_loader)

# Save models
torch.save(resnet_model.state_dict(), 'resnet_cifar10.pth')
torch.save(mobilenet_model.state_dict(), 'mobilenet_cifar10.pth')


# Function to quantize a model to INT8
def quantize_model(model, model_path):
    # Load the model
    torch.backends.quantized.engine = 'fbgemm'
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Fuse the layers for quantization
    model = torch.quantization.fuse_modules(model, [['conv1', 'bn1', 'relu']])

    # Prepare the model for static quantization
    model.qconfig = quantization.get_default_qconfig('fbgemm')
    #qconfig = quantization.get_default_qconfig('fbgemm')
    #model.layer1.qconfig = default_per_channel_qconfig
    #model.fc.qconfig = default_qconfig


    quantization.prepare(model, inplace=True)

    # Perform quantization-aware training (optional)
    # Simulate this with one inference pass
    for images, _ in train_loader:
        images = images.to(device)
        model(images)

    # Convert the model to quantized version
    #net_prepared = net_prepared.to('cpu')
    #net_prepared.eval()
    #net_int = torch.quantization.convert(net_prepared)
    quantization.convert(model, inplace=True)
    torch.save(model.state_dict(), f'quantized_{model_path}')

# Quantize both ResNet and MobileNet models
quantize_model(resnet_model, 'resnet_cifar10.pth')
quantize_model(mobilenet_model, 'mobilenet_cifar10.pth')


# Load the quantized models
def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Load quantized models
quantized_resnet = load_model(resnet_model, 'quantized_resnet_cifar10.pth')
quantized_mobilenet = load_model(mobilenet_model, 'quantized_mobilenet_cifar10.pth')

# Step 2: System Resource Monitoring Functions
def get_cpu_usage():
    """Get current system CPU usage."""
    return psutil.cpu_percent(interval=1)

def get_gpu_usage():
    """Get current GPU memory usage."""
    gpus = GPUtil.getGPUs()
    if gpus:
        return max([gpu.memoryUtil for gpu in gpus]) * 100  # Get memory utilization in percentage
    return None  # Return None if no GPU is available

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
        return quantized_mobilenet
    else:
        print("Using ResNet (heavier model)")
        return quantized_resnet


# Function for inference
def run_inference(model, input_data):
    with torch.no_grad():
        output = model(input_data)
        return F.softmax(output, dim=1)

# Preprocess input
def preprocess_input(image):
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
    return image.to(device)

# Run dynamic inference
for images, labels in test_loader:
    cpu_usage = get_cpu_usage()

    # Choose model based on CPU usage
    selected_model = choose_model(cpu_usage)
    
    # Preprocess input image
    input_image = preprocess_input(images[0])

    # Run inference
    output = run_inference(selected_model, input_image)
    print(f"Predicted: {torch.argmax(output)}")
