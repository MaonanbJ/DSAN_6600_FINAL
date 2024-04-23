import sys
import os
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import Mean, CategoricalAccuracy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Ensure the correct path
sys.path.append(os.getcwd())

# Import your model architecture and data reading function
from model.ResNet34 import resnet34
from detection.bin.cnndetection import read_data

# Set training parameters
im_height = 224
im_width = 224
num_classes = 4  # Update this with the actual number of classes
batch_size = 16
epochs = 20

# Load and process the data
X_train, X_val, X_test, Y_train, Y_val, Y_test = read_data()

# Create the ResNet34 model
model = resnet34(im_width=im_width, im_height=im_height, num_classes=num_classes)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0002),
              loss=CategoricalCrossentropy(from_logits=False),
              metrics=[CategoricalAccuracy()])

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, verbose=1),
    ModelCheckpoint(filepath='./save_weights/resNet_34.ckpt', save_best_only=True, monitor='val_loss', verbose=1)
]

# Train the model
history = model.fit(
    x=X_train, 
    y=Y_train, 
    batch_size=batch_size, 
    epochs=epochs, 
    validation_data=(X_val, Y_val),
    callbacks=callbacks
)

# Evaluate the model on the test dataset
test_loss, test_accuracy = model.evaluate(X_test, Y_test, batch_size=batch_size)
print(f'Final Test Loss: {test_loss}, Test Accuracy: {test_accuracy * 100}')
