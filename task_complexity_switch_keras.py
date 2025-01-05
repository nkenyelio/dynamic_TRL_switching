import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import psutil
import GPUtil

# Step 1: Define multiple pre-trained models with application-aware scheduling
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

        # Store model accuracy, confidence, and input size function (initialize to low values)
        self.model_accuracies = {
            'resnet': 0.5,  # Placeholder accuracy
            'mobilenet': 0.5
        }
        self.model_confidences = {
            'resnet': 0.5,  # Placeholder confidence
            'mobilenet': 0.5
        }
        self.model_input_size = {
            'resnet': lambda x: x * 0.9,  # Placeholder input size scaling function
            'mobilenet': lambda x: x * 0.7
        }

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

    def update_model_performance(self, model_name, accuracy, confidence):
        """Update the accuracy and confidence of the model after validation/test."""
        if model_name in self.model_accuracies:
            self.model_accuracies[model_name] = accuracy
        if model_name in self.model_confidences:
            self.model_confidences[model_name] = confidence

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

# Step 3: Resource Cost Function with Input Size, Confidence, and Task Complexity
def compute_resource_cost(accuracy, confidence, cpu_usage, gpu_usage, memory_usage, task_complexity, input_size, weights):
    """
    Compute the resource cost function with task complexity, input size, and inference confidence.

    :param accuracy: Model accuracy.
    :param confidence: Model's inference confidence.
    :param cpu_usage: Current CPU usage.
    :param gpu_usage: Current GPU usage.
    :param memory_usage: Current memory usage.
    :param task_complexity: The complexity of the current task.
    :param input_size: Size of the input (affects resource cost).
    :param weights: A dictionary with weight values for each component.
    :return: Computed cost.
    """
    alpha = weights['accuracy']
    beta = weights['cpu']
    gamma = weights['gpu']
    delta = weights['memory']
    epsilon = weights['confidence']
    zeta = weights['complexity']
    eta = weights['input_size']

    # Cost function: lower cost is better
    cost = (alpha * (1 - accuracy) +
            epsilon * (1 - confidence) +
            beta * cpu_usage +
            gamma * gpu_usage +
            delta * memory_usage +
            zeta * task_complexity +
            eta * input_size)

    return cost

# Step 4: Dynamic model switching with confidence, input size, and task complexity
def dynamic_model_switching_with_input_size(model_manager, criterion, optimizer, dataloader, val_loader, num_epochs=10, resource_limits=None, weights=None, task_complexity=1, application_input_size=1):
    """
    Perform dynamic model switching based on resource cost, inference confidence, input size, and task complexity.

    :param resource_limits: Dictionary containing resource thresholds.
    :param weights: Weights for accuracy, confidence, CPU, GPU, memory, complexity, and input size in the cost function.
    :param task_complexity: Complexity of the task (simple=1, complex=5).
    :param application_input_size: Input size of the application (small=1, large=5).
    """
    if resource_limits is None:
        resource_limits = {'cpu': 70, 'gpu': 80, 'memory': 75}  # Default limits

    if weights is None:
        weights = {'accuracy': 0.4, 'confidence': 0.3, 'cpu': 0.1, 'gpu': 0.1, 'memory': 0.1, 'complexity': 0.2, 'input_size': 0.2}  # Default weights

    for epoch in range(num_epochs):
        running_loss = 0.0

        for inputs, labels in dataloader:
            # Monitor system resources
            cpu_usage = get_cpu_usage()
            gpu_usage = get_gpu_usage()
            memory_usage = get_memory_usage()

            # Compute resource costs for each model
            for model_name, model in model_manager.models.items():
                accuracy = model_manager.model_accuracies[model_name]
                confidence = model_manager.model_confidences[model_name]
                input_size_factor = model_manager.model_input_size[model_name](application_input_size)
                cost = compute_resource_cost(accuracy, confidence, cpu_usage, gpu_usage, memory_usage, task_complexity, input_size_factor, weights)
                print(f"Model: {model_name}, Cost: {cost:.4f}")

                # Choose the model with the lowest cost
                if model_name == 'mobilenet' or model_name == 'resnet':  # Adjust conditions based on actual scenario
                    model_manager.switch_model(model_name)

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

        # Validate the model and update accuracy and confidence
        validate_model(model_manager, val_loader)

    print("Training complete")

# Step 5: Validation function to calculate model accuracy and confidence
def validate_model(model_manager, val_loader):
    """Validate the active model, calculate accuracy and confidence."""
    correct = 0
    total = 0
    total_confidence = 0.0
    model_manager.active_model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model_manager.forward(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Confidence based on softmax output
            softmax_outputs = torch.softmax(outputs, dim=1)
            confidence, _ = torch.max(softmax_outputs, dim=1)
            total_confidence += confidence.mean().item()

    accuracy = correct / total
    average_confidence = total_confidence / total
    active_model_name = list(model_manager.models.keys())[list(model_manager.models.values()).index(model_manager.active_model)]
    model_manager.update_model_performance(active_model_name, accuracy, average_confidence)
    print(f'Validation Accuracy of {active_model_name}: {accuracy:.4f}, Confidence: {average_confidence:.4f}')

# Step 6: Example training setup
def train_model_with_application_aware_switching():
    # Dummy dataset and dataloader (Replace with your actual dataset)
    from torch.utils.data import DataLoader, TensorDataset
    inputs = torch.randn(100, 3, 224, 224)  # 100 samples, 3 channels, 224x224 image size
    labels = torch.randint(0, 2, (100,))  # Binary classification example
    dataset = TensorDataset(inputs, labels)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

    # Validation dataset
    val_inputs = torch.randn(20, 3, 224, 224)
    val_labels = torch.randint(0, 2, (20,))
    val_dataset = TensorDataset(val_inputs, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False)

    # Initialize the model manager with 2 output classes
    model_manager = ModelManager(num_classes=2)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_manager.active_model.parameters(), lr=0.001)

    # Set task complexity (simple=1, complex=5)
    task_complexity = 3

    # Set application input size (small=1, large=5)
    application_input_size = 2

    # Train the model with dynamic switching
    dynamic_model_switching_with_input_size(model_manager, criterion, optimizer, dataloader, val_loader, task_complexity=task_complexity, application_input_size=application_input_size)

# Run the training with application-aware scheduling, input size, confidence, and complexity
if __name__ == "__main__":
    train_model_with_application_aware_switching()
