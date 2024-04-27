# -*- coding: utf-8 -*- 
# @Time : 2024/4/19 17:49
# @Author : Jerry Hao
# @File : VGG19train.py 
# @Desc :

import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from model.VGG19 import VGG19

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Function to walk through directories and fetch image paths
def get_image_paths(directory):
    image_paths = []
    for dirname, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):  # Add valid image extensions here
                image_paths.append(os.path.join(dirname, filename))
    return image_paths


# Function to find the minimum image count across categories
def find_minimum_image_count(categories):
    return min(len(images) for images in categories.values())


# Function to create a balanced dataset
def create_balanced_dataset(image_categories, min_count):
    balanced_dataset = {}
    for category_name, image_list in image_categories.items():
        if len(image_list) >= min_count:
            balanced_dataset[category_name] = random.sample(image_list, min_count)
        else:
            balanced_dataset[category_name] = image_list
    return balanced_dataset


# Define the directories
non_demented_dir = 'Data/Non Demented'
mild_demented_dir = 'Data/Mild Dementia'
moderate_demented_dir = 'Data/Moderate Dementia'
very_mild_demented_dir = 'Data/Very Mild Dementia'

# Get image paths
image_categories = {
    "Non Demented": get_image_paths(non_demented_dir),
    "Mild Dementia": get_image_paths(mild_demented_dir),
    "Moderate Dementia": get_image_paths(moderate_demented_dir),
    "Very Mild Dementia": get_image_paths(very_mild_demented_dir)
}

# Create balanced dataset
min_image_count = find_minimum_image_count(image_categories)
balanced_dataset = create_balanced_dataset(image_categories, min_image_count)

# Prepare data and labels
X_data = []
Y_data = []
for category, images in balanced_dataset.items():
    X_data.extend(images)
    Y_data.extend([category] * len(images))

# Encode labels
label_encoder = LabelEncoder()
Y_encoded = label_encoder.fit_transform(Y_data)

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_encoded, test_size=0.2, random_state=42)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.25, random_state=42)


# Define a dataset class for PyTorch
class DementiaDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[index]

        if self.transform:
            image = self.transform(image)

        return image, label


# Define transformations for data augmentation
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # VGGNet expects 224x224 sized images
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

# Define VGGNet-19 model
model = VGG19()
for param in model.parameters():
    param.requires_grad = False

# Modify the classifier layer
num_features = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_features, 4)  # Adjust output classes to 4

# Move model to device
model = model.to(device)

# Initialize the loss criterion and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Train the model
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
        print('Epoch: {}, Loss: {}'.format(epoch + 1, total_loss / len(train_loader)))
    return model


trained_model = train_model(model, criterion, optimizer, train_loader)


# Evaluation function
def evaluate_model(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy


# Evaluate the model
val_accuracy = evaluate_model(trained_model, val_loader)
test_accuracy = evaluate_model(trained_model, test_loader)
print(f'Validation Accuracy: {val_accuracy}%')
print(f'Test Accuracy: {test_accuracy}%')

from sklearn.metrics import classification_report


# Evaluation function with prediction outputs
def evaluate_model_with_output(model, loader):
    model.eval()
    predictions = []
    true_labels = []
    with torch.no_grad():
        for data, labels in loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    return predictions, true_labels


# Get predictions and true labels for validation and test sets
val_predictions, val_true_labels = evaluate_model_with_output(trained_model, val_loader)
test_predictions, test_true_labels = evaluate_model_with_output(trained_model, test_loader)

# Generate classification report
val_classification_report = classification_report(val_true_labels, val_predictions)
test_classification_report = classification_report(test_true_labels, test_predictions)

print("Validation Classification Report:")
print(val_classification_report)
print("Test Classification Report:")
print(test_classification_report)
