import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import psutil
import GPUtil

# Step 1: Define multiple pre-trained models
class ModelManager:
    def __init__(self, num_classes):
        # Load multiple models with different resource footprints
        self.models = {
            'resnet': models.resnet18(pretrained=True),
            'mobilenet': models.mobilenet_v2(pretrained=True)
        }

        # Modify the final layers to match the number of classes for the task
        self.models['resnet'].fc = nn.Linear(self.models['resnet'].fc.in_features, num_classes)
        self.models['mobilenet'].classifier[1] = nn.Linear(self.models['mobilenet'].classifier[1].in_features, num_classes)

        # Set the current active model (default to resnet)
        self.active_model = self.models['resnet']

    def switch_model(self, model_name):
        """Switch the active model based on the provided model name."""
        if model_name in self.models:
            self.active_model = self.models[model_name]
            print(f"Switched to {model_name}")
        else:
            print(f"Model {model_name} not found!")

    def forward(self, x):
        """Forward pass through the active model."""
        return self.active_model(x)

# Step 2: System Resource Monitoring Functions
def get_cpu_usage():
    """Get current system CPU usage."""
    return psutil.cpu_percent()

def get_gpu_usage():
    """Get current GPU memory usage."""
    gpus = GPUtil.getGPUs()
    if gpus:
        return max([gpu.memoryUtil for gpu in gpus]) * 100  # Get memory utilization in percentage
    return None  # Return None if no GPU is available

def get_memory_usage():
    """Get current system memory usage."""
    return psutil.virtual_memory().percent

# Step 3: Dynamic resource-aware model switching logic
def dynamic_model_switching(model_manager, criterion, optimizer, dataloader, num_epochs=10, resource_limits=None):
    """
    Perform dynamic model switching based on resource usage.

    :param resource_limits: Dictionary containing resource thresholds to trigger model switching.
                            Example: {'cpu': 70, 'gpu': 80, 'memory': 75}
    """
    if resource_limits is None:
        resource_limits = {'cpu': 70, 'gpu': 80, 'memory': 75}  # Default limits

    for epoch in range(num_epochs):
        running_loss = 0.0

        for inputs, labels in dataloader:
            # Monitor system resources
            cpu_usage = get_cpu_usage()
            gpu_usage = get_gpu_usage()
            memory_usage = get_memory_usage()

            # Check if any resource exceeds the threshold
            if (cpu_usage > resource_limits['cpu'] or
                (gpu_usage and gpu_usage > resource_limits['gpu']) or
                memory_usage > resource_limits['memory']):
                model_manager.switch_model('mobilenet')  # Switch to lighter model
            else:
                model_manager.switch_model('resnet')  # Switch to heavier model

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model_manager.forward(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader)}")

    print("Training complete")

# Step 4: Example training setup
def train_model_with_resource_aware_switching():
    # Dummy dataset and dataloader (Replace with your actual dataset)
    from torch.utils.data import DataLoader, TensorDataset
    inputs = torch.randn(100, 3, 224, 224)  # 100 samples, 3 channels, 224x224 image size
    labels = torch.randint(0, 2, (100,))  # Binary classification example
    dataset = TensorDataset(inputs, labels)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

    # Initialize the model manager with 2 output classes
    model_manager = ModelManager(num_classes=2)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_manager.active_model.parameters(), lr=0.001)

    # Train the model with dynamic switching
    dynamic_model_switching(model_manager, criterion, optimizer, dataloader)

# Run the training with resource-aware scheduling
if __name__ == "__main__":
    train_model_with_resource_aware_switching()
