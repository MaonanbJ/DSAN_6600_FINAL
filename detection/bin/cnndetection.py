import sys
import os
sys.path.append(os.getcwd())
from detection.utils.preprocessing import DementiaImageDataset

def read_data():
    # Define the paths to your data directories
    directories = {
        'Mild Dementia': 'Data/Mild_Dementia_Choice',
        'Moderate Dementia': 'Data/Moderate_Dementia_Choice',
        'Non Demented': 'Data/Non_Dementia_Choice',
        'Very mild Dementia': 'Data/Very_Mild_Dementia_Choice'
    }
    test_directories = {
        'Mild Dementia': 'Data/Mild_Dementia_Test',
        'Moderate Dementia': 'Data/Moderate_Dementia',
        'Non Demented': 'Data/Non_Dementia_Test',
        'Very mild Dementia': 'Data/Very_Mild_Dementia_Test'
    }
    # directories = {'Mild Dementia': 'Dataset/Mild_Demented',
    #    'Moderate Dementia': 'Dataset/Moderate_Demented',
    #    'Non Demented': 'Dataset/Non_Demented',
    #    'Very mild Dementia': 'Dataset/Very_Mild_Demented'
    #}
    # Initialize the dataset processing class
    dataset_processor = DementiaImageDataset(directories)

    # Load and process the images
    dataset_processor.load_and_process_images()

    # Split the dataset
    # X_train, X_val, X_test, Y_train, Y_val, Y_test = dataset_processor.split_data()
    X_train, X_val, Y_train, Y_val = dataset_processor.split_data()

    test_processor = DementiaImageDataset(test_directories)

    test_processor.load_and_process_images()
    
    X_test, Y_test = test_processor.split_data(split=False)

    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Testing data shape: {X_test.shape}")
    print(f"Training labels shape: {Y_train.shape}")
    print(f"Validation labels shape: {Y_val.shape}")
    print(f"Testing labels shape: {Y_test.shape}")

    #return X_train, X_val, Y_train, Y_val
    return X_train, X_val, X_test, Y_train, Y_val, Y_test


