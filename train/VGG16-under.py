# -*- coding: utf-8 -*- 
# @Time : 2024/4/19 17:32
# @Author : Jerry Hao
# @File : VGG16-under.py 
# @Desc :

import os
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from model.VGG16 import Vgg

# read data
non_demented_dir = 'Data/Non Demented'
mild_demented_dir = 'Data/Mild Dementia'
moderate_demented_dir = 'Data/Moderate Dementia'
very_mild_demented_dir = 'Data/Very Mild Dementia'


def get_image_paths(directory):
    image_paths = []
    for dirname, _, filenames in os.walk(directory):
        for filename in filenames:
            image_paths.append(os.path.join(dirname, filename))
    return image_paths


non_demented = get_image_paths(non_demented_dir)
mild_demented = get_image_paths(mild_demented_dir)
moderate_demented = get_image_paths(moderate_demented_dir)
very_mild_demented = get_image_paths(very_mild_demented_dir)

image_categories = {
    "Non Demented": non_demented,
    "Mild Demented": mild_demented,
    "Moderate Demented": moderate_demented,
    "Very Mild Demented": very_mild_demented
}


def find_minimum_image_count(categories):
    return min(len(images) for images in categories.values())


min_image_count = find_minimum_image_count(image_categories)


def create_balanced_dataset(categories, min_count):
    balanced_dataset = {}
    for category_name, image_list in categories.items():
        balanced_dataset[category_name] = random.sample(image_list, min_count)
    return balanced_dataset


balanced_dataset = create_balanced_dataset(image_categories, min_image_count)

X_data = []
Y_data = []

for category_name, images in balanced_dataset.items():
    label = category_name
    for image_path in images:
        try:
            img = Image.open(image_path)
            img = img.resize((128, 128))
            img = img.convert('RGB')
            img_array = np.array(img)
            X_data.append(img_array)
            Y_data.append(label)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

X_data = np.array(X_data)
Y_data = np.array(Y_data)

# data augmentation
augmentation = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.2, random_state=42)
X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, test_size=0.5, random_state=42)

# encode label
label_encoder = LabelEncoder()
Y_train_encoded = label_encoder.fit_transform(Y_train)
Y_val_encoded = label_encoder.transform(Y_val)
Y_test_encoded = label_encoder.transform(Y_test)

train_augmented = augmentation.flow(X_train, Y_train_encoded, batch_size=32)

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Vgg.tuner.search(train_augmented, epochs=10, validation_data=(X_val, Y_val_encoded))

best_hps = Vgg.tuner.get_best_hyperparameters()[0]

# train
best_model = Vgg.tuner.hypermodel.build(best_hps)
history = best_model.fit(train_augmented, epochs=10, validation_data=(X_val, Y_val_encoded))

# eveluation
test_loss, test_accuracy = best_model.evaluate(X_test, Y_test_encoded)
print(f"Test Accuracy: {test_accuracy}")

# plot
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Accuracy / Loss')
plt.ylim([0, 3])
plt.legend(loc='lower right')
plt.show()

test_loss, test_accuracy = best_model.evaluate(X_train, Y_train_encoded)
print(f"Test Accuracy: {test_accuracy}")

Y_pred = best_model.predict(X_test)
Y_pred_classes = np.argmax(Y_pred, axis=1)

print("Classification Report:")
print(classification_report(Y_test_encoded, Y_pred_classes))

conf_mat = confusion_matrix(Y_test_encoded, Y_pred_classes)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()
