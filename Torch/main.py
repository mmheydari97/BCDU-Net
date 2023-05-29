import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from BCDUNet import BCDUNet

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
input_dim = 3
output_dim = 3
num_filter = 64
frame_size = (256, 256)
bidirectional = False
norm = 'instance'
batch_size = 4
epochs = 5
learning_rate = 0.001

# Load the dataset
transform = transforms.Compose([
    transforms.Resize(frame_size),
    transforms.ToTensor()
])

train_data = datasets.Cityscapes('./data', split='train', mode='fine', target_type='semantic', transform=transform, target_transform=transform)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# Initialize the model
model = BCDUNet(input_dim, output_dim, num_filter, frame_size, bidirectional, norm).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Move data and targets to the device
        data = data.to(device)
        targets = targets.to(device)

        # Forward pass
        scores = model(data)
        loss = criterion(scores, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Update weights
        optimizer.step()

        # Print progress
        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item()}")

print("Training completed.")