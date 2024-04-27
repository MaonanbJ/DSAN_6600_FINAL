# -*- coding: utf-8 -*- 
# @Time : 2024/4/16 16:20
# @Author : Jerry Hao
# @File : CNN.py 
# @Desc :

import os
import random

import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset


# Function to walk through directories and fetch image paths
def get_image_paths(directory):
    image_paths = []
    for dirname, _, filenames in os.walk(directory):
        for filename in filenames:
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
    "Mild Demented": get_image_paths(mild_demented_dir),
    "Moderate Demented": get_image_paths(moderate_demented_dir),
    "Very Mild Demented": get_image_paths(very_mild_demented_dir)
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


# Define a simple neural network model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 32 * 32)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return self.log_softmax(x)
