import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from BCDUNet import BCDUNet
import matplotlib.pyplot as plt


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
input_dim = 3
output_dim = 1
num_filter = 64
frame_size = (256, 256)
bidirectional = False
norm = 'instance'
batch_size = 4
epochs = 5
learning_rate = 0.001
vis_cmap = 'gray' if output_dim == 1 else 'rgb'

# Load the dataset
transform = transforms.Compose([
    transforms.Resize(frame_size),
    transforms.ToTensor()
])

train_data = datasets.VOCSegmentation('./data', year='2012', image_set='train', transform=transform, target_transform=transform)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

eval_data = datasets.VOCSegmentation('./data', year='2012', image_set='val', transform=transform, target_transform=transform)
eval_loader = DataLoader(eval_data, batch_size=batch_size, shuffle=False)


# Initialize the model
model = BCDUNet(input_dim, output_dim, num_filter, frame_size, bidirectional, norm).to(device)

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def visualize_samples(inputs, generated_outputs, ground_truth, filename):
    num_samples = inputs.size(0)
    fig, axes = plt.subplots(num_samples, 3, figsize=(10, 10))
    fig.tight_layout()

    for i in range(num_samples):
        axes[i, 0].imshow(inputs[i].permute(1, 2, 0))
        axes[i, 0].set_title('Input')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(generated_outputs[i].squeeze(), cmap=vis_cmap)
        axes[i, 1].set_title('Generated Output')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(ground_truth[i].permute(1, 2, 0), cmap=vis_cmap)
        axes[i, 2].set_title('Ground Truth')
        axes[i, 2].axis('off')

    plt.savefig(filename)

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
        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item()}")


torch.save(model.state_dict(), "model.pth")
print("Training completed.")


# Evaluation set loader
eval_data = datasets.VOCSegmentation('./data', year='2012', image_set='val', transform=transform, target_transform=transform)
eval_loader = DataLoader(eval_data, batch_size=batch_size, shuffle=False)

# Evaluation loop
model.eval()
total_loss = 0

with torch.no_grad():
    for batch_idx, (data, targets) in enumerate(eval_loader):
        data = data.to(device)
        targets = targets.to(device)
        scores = model(data)
        loss = criterion(scores, targets)
        total_loss += loss.item()

        # Visualize sample triplets
        if batch_idx == 0:
            visualize_samples(data.cpu(), torch.sigmoid(scores).cpu(), targets.cpu(), f'fig_{batch_idx}.jpg')

avg_loss = total_loss / len(eval_loader)
print(f"Evaluation Loss: {avg_loss}")



