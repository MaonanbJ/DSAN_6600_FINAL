# -*- coding: utf-8 -*- 
# @Time : 2024/4/16 18:22
# @Author : Jerry Hao
# @File : CNNtrain.py 
# @Desc :
from model.CNN import SimpleCNN,DementiaDataset
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset


# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transformations for data augmentation
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Data loaders for PyTorch
train_dataset = DementiaDataset(X_train, Y_train, transform=transform)
val_dataset = DementiaDataset(X_val, Y_val, transform=transform)
test_dataset = DementiaDataset(X_test, Y_test, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize the model, loss criterion, and optimizer
model = SimpleCNN(num_classes=4).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train_model(model, criterion, optimizer, train_loader, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print('Epoch: {}, Loss: {}'.format(epoch+1, total_loss / len(train_loader)))
    return model

