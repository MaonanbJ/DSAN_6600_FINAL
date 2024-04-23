import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

class DementiaImageDataset:
    def __init__(self, directories, size=(224, 224)):
        self.directories = directories
        self.size = size
        self.image_data = []
        self.encoded_labels = []
        self.category_labels = {category: i for i, category in enumerate(self.directories.keys())}
        print("Category labels:", self.category_labels)
    
    def get_image_paths(self, directory):
        image_paths = []
        for dirname, _, filenames in os.walk(directory):
            for filename in filenames:
                image_paths.append(os.path.join(dirname, filename))
        return image_paths

    def one_hot_encode(self, label, num_labels):
        one_hot = np.zeros(num_labels)
        one_hot[label] = 1
        return one_hot

    def process_image(self, image_path):
        try:
            with Image.open(image_path) as img:
                img = img.resize(self.size)
                img = img.convert('RGB') 
                img_arr = np.asarray(img)/255.0
                return img_arr
        except IOError as e:
            print(f"Could not open image {image_path}: {e}")
            return None

    def load_and_process_images(self):
        for category_name, directory in self.directories.items():
            print(f"Processing {category_name}...")
            label = self.category_labels[category_name]
            images = self.get_image_paths(directory)
            for image_path in images:
                img = self.process_image(image_path)
                if img is not None:
                    self.image_data.append(img)
                    encoded_label = self.one_hot_encode(label, len(self.category_labels))
                    self.encoded_labels.append(encoded_label)

    def split_data(self, test_size=0.2, val_size=0.5, split = True):
        X = np.array(self.image_data)
        Y = np.array(self.encoded_labels)
        #print("Data shape:", X.shape)
        # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=42)
        #X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, test_size=val_size, random_state=42)
        if split:
            X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=test_size, random_state=42)
            return X_train, X_val, Y_train, Y_val
        else:
            return X, Y

