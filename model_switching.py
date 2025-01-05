import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

# Step 1: Define multiple pre-trained models
class ModelManager:
    def __init__(self, num_classes):
        # Load multiple models (you can add more)
        self.models = {
            'resnet': models.resnet18(pretrained=True),
            'vgg': models.vgg16(pretrained=True)
        }

        # Modify the final layers to match the number of classes for the task
        self.models['resnet'].fc = nn.Linear(self.models['resnet'].fc.in_features, num_classes)
        self.models['vgg'].classifier[6] = nn.Linear(self.models['vgg'].classifier[6].in_features, num_classes)

        # Store the current active model (default to resnet)
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

# Step 2: Example training setup
def train_model(model_manager, dataloader, criterion, optimizer, num_epochs=10):
    model_manager.active_model.train()  # Set the current model to training mode

    for epoch in range(num_epochs):
        running_loss = 0.0

        for inputs, labels in dataloader:
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

# Step 3: Switch models dynamically based on some condition
def dynamic_model_switching(model_manager, criterion, optimizer, dataloader, switch_criterion='loss_threshold'):
    loss_threshold = 0.3  # Define your condition

    # Example training loop that switches models dynamically
    for epoch in range(5):  # Use a small number of epochs for demo
        total_loss = 0.0

        # Perform training on the current model
        for inputs, labels in dataloader:
            outputs = model_manager.forward(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

        average_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch + 1}: Avg Loss = {average_loss:.4f}')

        # Dynamic switching logic
        if average_loss < loss_threshold and model_manager.active_model == model_manager.models['resnet']:
            model_manager.switch_model('vgg')  # Switch to a different model
        elif average_loss >= loss_threshold and model_manager.active_model == model_manager.models['vgg']:
            model_manager.switch_model('resnet')  # Switch back to another model

# Example of how to use this setup
if __name__ == "__main__":
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

    # Train and dynamically switch models
    dynamic_model_switching(model_manager, criterion, optimizer, dataloader)
