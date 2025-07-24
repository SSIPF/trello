import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters
batch_size = 32
num_epochs = 12
learning_rate = 0.001
img_size = 224

# Transforms
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
])

# Dataset
dataset = datasets.ImageFolder(root=".", transform=transform)  # expects ./group1, ./group2
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Class names (should be ['group1', 'group2'])
classes = dataset.classes
print("Classes:", classes)

# Simple CNN
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> 112x112

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> 56x56

            nn.Conv2d(32, 64, 3, padding=1),  # <--- added layer
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> 28x28
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 100),  # adjusted input features
            nn.ReLU(),
            nn.Linear(100, 2)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


model = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()

    acc = correct / len(dataset)
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Accuracy: {acc:.4f}")

# Save model
torch.save(model.state_dict(), "../ml_sk3/drone_classifier.pth")
