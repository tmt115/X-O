import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm
import numpy as np
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm

class XODataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = ImageFolder(data_dir, transform=transform)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @property
    def classes(self):
        return self.data.classes

    @property
    def class_to_idx(self):
        return self.data.class_to_idx

data_dir = "C:/Users/miket/CEED/dataset/train"
labels = {k: v for k, v in ImageFolder(data_dir).class_to_idx.items()}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

dataset = XODataset(data_dir, transform)

image, label = dataset[500]

dataloader = DataLoader(dataset, batch_size = 32, shuffle = True)

class XOClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.model = timm.create_model(
            "efficientnet_b0",
            pretrained=True,
            num_classes=num_classes
        )

    def forward(self, x):
        return self.model(x)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = XOClassifier(num_classes=3).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.0001)

train_folder = "C:/Users/miket/CEED/dataset/train"
val_folder = "C:/Users/miket/CEED/dataset/val"
test_folder = "C:/Users/miket/CEED/dataset/test"

train_dataset = XODataset(train_folder, transform = transform)
val_dataset = XODataset(val_folder, transform = transform)
test_dataset = XODataset(test_folder, transform = transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle = True)

num_epoch = 5
train_losses, val_losses = [], []

for epoch in range(num_epoch):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader, desc='Training loop'):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * labels.size(0)
    training_loss = running_loss/len(train_loader.dataset)
    train_losses.append(training_loss)
    
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Validation loop'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * labels.size(0)
        val_loss = running_loss / len(val_loader.dataset)
        val_losses.append(val_loss)

    print(f"Epoch {epoch+1}/{num_epoch} - Train loss: {training_loss}, Validation loss: {val_loss}")