# -*- coding: utf-8 -*- 
# @Time : 2024/4/26 17:47 
# @Author : Jerry Hao
# @File : VGG16-overTen.py 
# @Desc :

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from imblearn.over_sampling import SMOTE
from kerastuner import RandomSearch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical


# Modified function to load and balance images
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
                image = cv2.resize(image, (256, 256))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                category_data.append(image)
                category_labels.append(class_num)
        category_data = np.array(category_data)
        category_labels = np.array(category_labels)
        if len(np.unique(category_labels)) > 1:  # Ensuring SMOTE applicability
            category_data, category_labels = smote.fit_resample(category_data.reshape(len(category_data), -1),
                                                                category_labels)
            category_data = category_data.reshape(-1, 256, 256, 3)  # Reshaping back to image dimensions
        data.extend(category_data)
        labels.extend(category_labels)
    data = np.array(data)
    labels = np.array(labels)
    return data, to_categorical(labels, num_classes=4)


# Load and prepare data
base_path = 'Data'
data, labels = load_images_and_labels(base_path)
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Convert the train data to generator
train_datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = train_datagen.flow(X_train, y_train, batch_size=32)


# Build the model for RandomSearch
def build_model(hp):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
    for layer in base_model.layers:
        layer.trainable = False

    x = Flatten()(base_model.output)
    x = Dense(units=hp.Int('units', min_value=256, max_value=1024, step=256), activation='relu')(x)
    x = Dropout(rate=hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1))(x)
    predictions = Dense(4, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Model tuning configuration
tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=10,
    executions_per_trial=1,
    directory='my_dir',
    project_name='VGG16_tuning'
)

# Model search
tuner.search(train_generator, steps_per_epoch=len(X_train) // 32, epochs=10, validation_data=(X_test, y_test))

# Get the best model
best_model = tuner.get_best_models(num_models=1)[0]

# Fit the model
callbacks = [EarlyStopping(monitor='val_loss', patience=3, verbose=1)]
history = best_model.fit(train_generator, steps_per_epoch=len(X_train) // 32, epochs=20,
                         validation_data=(X_test, y_test), callbacks=callbacks)


# Plot training history
def plot_training_history(history):
    plt.figure(figsize=(14, 7))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()


plot_training_history(history)

# Evaluation and Reporting
evaluation = best_model.evaluate(X_test, y_test)
print("Evaluation Results:", evaluation)

predictions = best_model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test, axis=1)
conf_matrix = confusion_matrix(true_classes, predicted_classes)
class_report = classification_report(true_classes, predicted_classes,
                                     target_names=['Mild Dementia', 'Moderate Dementia', 'Non Demented',
                                                   'Very Mild Dementia'])

print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)
