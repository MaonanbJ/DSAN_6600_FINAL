import sys
import os
sys.path.append(os.getcwd())
from detection.utils.preprocessing import DementiaImageDataset

def read_data():
    # Define the paths to your data directories
    directories = {
        'Mild Dementia': './Data/Mild Dementia',
        'Moderate Dementia': './Data/Moderate Dementia',
        'Non Demented': './Data/Non Demented',
        'Very mild Dementia': './Data/Very mild Dementia'
    }

    # Initialize the dataset processing class
    dataset_processor = DementiaImageDataset(directories)

    # Load and process the images
    dataset_processor.load_and_process_images()

    # Split the dataset
    X_train, X_val, X_test, Y_train, Y_val, Y_test = dataset_processor.split_data()

    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Testing data shape: {X_test.shape}")
    print(f"Training labels shape: {Y_train.shape}")
    print(f"Validation labels shape: {Y_val.shape}")
    print(f"Testing labels shape: {Y_test.shape}")

    return X_train, X_val, X_test, Y_train, Y_val, Y_test


