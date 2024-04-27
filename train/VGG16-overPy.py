# -*- coding: utf-8 -*- 
# @Time : 2024/4/20 17:45
# @Author : Jerry Hao
# @File : VGG16-overPy.py 
# @Desc :

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models


# Custom dataset class
class ImageDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index]
        label = self.labels[index]
        if self.transform:
            image = self.transform(image)
        return image, label


# Load and preprocess data
def load_images_and_labels(base_path):
    categories = ['Mild Dementia', 'Moderate Dementia', 'Non Demented', 'Very Mild Dementia']
    data = []
    labels = []
    smote = SMOTE()
    for category in categories:
        path = os.path.join(base_path, category)
        class_num = categories.index(category)
        category_data = []
        category_labels = []
        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            image = cv2.imread(img_path)
            if image is not None:
                image = cv2.resize(image, (224, 224))  # Adjust size for VGG19
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                category_data.append(image)
                category_labels.append(class_num)
        category_data = np.array(category_data)
        category_labels = np.array(category_labels)
        if len(np.unique(category_labels)) > 1:
            category_data, category_labels = smote.fit_resample(category_data.reshape(len(category_data), -1),
                                                                category_labels)
            category_data = category_data.reshape(-1, 224, 224, 3)
        data.extend(category_data)
        labels.extend(category_labels)
    data = np.array(data)
    labels = np.array(labels)
    return data, torch.tensor(labels, dtype=torch.long)


# Load data
base_path = 'Data'
data, labels = load_images_and_labels(base_path)
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Data augmentation and normalization
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dataset and DataLoader
train_dataset = ImageDataset(X_train, y_train, transform=train_transform)
test_dataset = ImageDataset(X_test, y_test, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Model setup
model = models.vgg19(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

# Modify the classifier
model.classifier[6] = nn.Linear(model.classifier[6].in_features, 4)

# Wrap model with DataParallel for multi-GPU training
model = nn.DataParallel(model)  # This line is crucial for using multiple GPUs

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.module.classifier.parameters(), lr=0.001)  # Adjust for DataParallel


# Early stopping setup
class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


# Training function with history and early stopping
def train_model(model, train_loader, test_loader, criterion, optimizer, epochs=10, early_stopping=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    history = {'train_loss': [], 'train_acc': []}
    val_loss_min = np.Inf
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        print(f'Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f}')

        # Check for early stopping
        if early_stopping:
            early_stopping(epoch_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
    return model, history


# Implement early stopping
early_stopping = EarlyStopping(patience=5, verbose=True)

# Train the model
trained_model, history = train_model(model, train_loader, test_loader, criterion, optimizer, epochs=10,
                                     early_stopping=early_stopping)


# Plot training history
def plot_training_history(history):
    plt.figure(figsize=(14, 7))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.title('Training Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.show()


plot_training_history(history)


# Evaluate the model
def evaluate_model(model, test_loader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)
    y_pred = []
    y_true = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())
    return y_true, y_pred


labels, preds = evaluate_model(trained_model, test_loader)
print(classification_report(labels, preds,
                            target_names=['Mild Dementia', 'Moderate Dementia', 'Non Demented', 'Very Mild Dementia']))

# Generate the confusion matrix
conf_matrix = confusion_matrix(labels, preds)
print("Confusion Matrix:")
print(conf_matrix)
