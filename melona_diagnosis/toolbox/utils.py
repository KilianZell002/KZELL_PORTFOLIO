import os
import time
import zipfile
import csv
import shutil
import random
import re

import torch
import torch.nn as nn
import torch.nn.functional as Fu
import torchvision.transforms.functional as F

import numpy as np

from toolbox import dataset

'''
---------------------------------------------------------------------------------------------
Data Loading and Processsing
---------------------------------------------------------------------------------------------
'''
def unzip_data(DATASET_PATH, DATASET_USED):
    """
    Unzips a dataset file into the specified directory.

    Args:
    - DATASET_PATH (str): Path to the directory where the dataset file is located.
    - DATASET_USED (str): Name of the dataset file without the extension.

    Returns:
    - None: Returns None if an error occurs during the extraction process.
    """

    # Create the file path by concatenating 'path', 'data', and ".zip"
    file_path = os.path.join(DATASET_PATH, f'{DATASET_USED}.zip')
    
    try:
        # Check if the file exists
        if os.path.exists(file_path):
            print(f"The file {file_path} is properly loaded.")
        else:
            print(f"The file {file_path} is not properly loaded.")

        # Record the start time for measuring execution time
        start_time = time.time()

        # Open and extract the contents of the ZIP file to a temporary folder
        zip_ref = zipfile.ZipFile(file_path, 'r')
        zip_ref.extractall(DATASET_PATH)
        zip_ref.close()

        # Record the end time and calculate execution time
        end_time = time.time()
        execution_time = end_time - start_time

        # Print a success message and the execution time
        print('Dataset successfully uploaded to the current directory.')
        print(f'Execution time: {execution_time} seconds')
    
    except Exception as e:
        # Handle any exceptions that might occur during file existence check or extraction
        print(f"An error occurred: {str(e)}")
        return None

def process_folder_data(input_folder, output_folder, dataset='HAM10000'):
    """
    Process data files in the input folder, rename them, and move them to the output folder.

    Args:
    - input_folder (str): Path to the folder containing data files.
    - output_folder (str): Path to the folder where processed files will be moved.
    - dataset (str): Dataset identifier. Default is 'HAM10000'.

    Returns:
    - None
    """
    # Get a list of files with the '.jpg' extension in the input folder
    files = [f for f in os.listdir(input_folder) if f.endswith('.jpg')]
    files.sort(key=numerical_sort)  # Sort the files chronologically based on their names

    for index, file in enumerate(files):
        # Check if the file has a '.png' extension
        if file.lower().endswith('.jpg'):
            # Create a new filename with '.png' extension and the index
            new_filename = f"{index}.png"
            new_filepath = os.path.join(output_folder, new_filename)
            current_filepath = os.path.join(input_folder, file)

            # Rename the file by moving it to the output folder with the new name
            os.rename(current_filepath, new_filepath)
        else:
            # Remove non-JPG files
            os.remove(os.path.join(input_folder, file))

def process_folder_mask(input_folder, output_folder, dataset='HAM1000'):
    """
    Process mask files in the input folder, rename them, and move them to the output folder.

    Args:
    - input_folder (str): Path to the folder containing mask files.
    - output_folder (str): Path to the folder where processed mask files will be moved.
    - dataset (str): Dataset identifier. Default is 'HAM10000'.

    Returns:
    - None
    """
    # Get a list of files with the '.png' extension in the input folder
    files = [f for f in os.listdir(input_folder) if f.endswith('.png')]
    files.sort(key=numerical_sort)  # Sort the files chronologically based on their names

    for index, file in enumerate(files):
        if file.lower().endswith('.png'):
            base_name, _ = os.path.splitext(file)

            if dataset == 'ISIC2016':
                base_name = base_name.replace('_Segmentation', '')
            elif dataset == 'ISIC2017' or dataset == 'ISIC2018'  or dataset == 'HAM10000' :
                base_name = base_name.replace('_segmentation', '')

            # Create a new filename with '.png' extension and the index
            new_filename = f"{index}.png"
            new_filepath = os.path.join(output_folder, new_filename)
            current_filepath = os.path.join(input_folder, file)

            # Rename the file by moving it to the output folder with the new name
            os.rename(current_filepath, new_filepath)  # Rename the file
        else:
            # Remove non-PNG files
            os.remove(os.path.join(input_folder, file))

def process_file_label(input_file, output_file, dataset='HAM10000'):
    """
    Process a CSV file containing ISIC ID and labels, assigning an index based on the sorted numeric part of ISIC ID.
    
    Args:
    - input_file: Path to the input CSV file.
    - output_file: Path to the output CSV file.
    - dataset (str): Dataset identifier. Default is 'HAM10000'.
    
    Returns:
    - None
    """
    # Read the CSV file and store data in a list
    data = []
    with open(input_file, 'r') as file:
        reader = csv.reader(file, delimiter=',')  # Change delimiter to ','
        for row in reader:
            data.append(row)

    # Remove the header
    data = data[1:]
    
    # Sort the data based on the numeric part of ISIC ID
    data.sort(key=lambda x: int(x[0].split('_')[1]))

    # Process and update the data
    processed_data = []
    index = 0

    for row in data:
        # Get label
        label = row[1]

        # Replace labels with '1' for 'malignant' and '0' for 'benign'
        if label == 'malignant' or label == 'benign':
            label = 1 if label.lower() == 'malignant' else 0

        # Convert label to float
        label = float(label)

        if label == 1 or label == 0:
          # Append the processed data to the new list
          processed_data.append([index, label])

        # Update the index for the ID prefix
        index += 1

    # Write the processed data to the output file
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file, delimiter='\t')  # Change delimiter to '\t'
        writer.writerows(processed_data)
        
def format_dataset(PATH, task='segmentation', test_folder=False, benchmark=False):
    """
    Format the dataset by processing data and mask folders.

    Args:
        PATH (str): Path to the main directory containing 'data_train', 'data_test', 'gt_train', and 'gt_test' folders.
        task (str): Task identifier. Default is 'segmentation'.
        test_folder (bool): Flag indicating whether the dataset includes a 'test' folder. Default is False.
        benchmark (bool): Flag indicating whether the dataset uses benchmark segmentation. Default is False.

    Returns:
        None
    """
    if task == 'segmentation':
        # Process the 'data' folders for both training and testing
        process_folder_data(os.path.join(PATH, 'data_train'), os.path.join(PATH, 'data_train'), dataset=dataset)
        process_folder_data(os.path.join(PATH, 'data_test'), os.path.join(PATH, 'data_test'), dataset=dataset)
        
        # Process the 'gt' folders for both training and testing (masks)
        process_folder_mask(os.path.join(PATH, 'gt_train'), os.path.join(PATH, 'gt_train'), dataset=dataset)
        process_folder_mask(os.path.join(PATH, 'gt_test'), os.path.join(PATH, 'gt_test'), dataset=dataset)
    
    elif task == 'classification':
        if test_folder == True:
            # Process the 'data' folders for both training and testing
            process_folder_data(os.path.join(PATH, 'data_train'), os.path.join(PATH, 'data_train'), dataset=dataset)
            process_folder_data(os.path.join(PATH, 'data_test'), os.path.join(PATH, 'data_test'), dataset=dataset)
            
            # Process the 'gt' .cvs file for both training and testing (labels)
            process_file_label(os.path.join(PATH, 'gt_train.csv'), os.path.join(PATH, 'gt_train.csv'), dataset=dataset)
            process_file_label(os.path.join(PATH, 'gt_test.csv'), os.path.join(PATH, 'gt_test.csv'), dataset=dataset)

            if benchmark == True:
                # Process the 'gt' folders for both training and testing
                process_folder_mask(os.path.join(PATH, 'gt_train'), os.path.join(PATH, 'gt_train'), dataset=dataset)
                process_folder_mask(os.path.join(PATH, 'gt_test'), os.path.join(PATH, 'gt_test'), dataset=dataset)
        else:
            # Process the 'data' folders for both training and testing
            process_folder_data(os.path.join(PATH, 'data_train'), os.path.join(PATH, 'data_train'), dataset=dataset)

            # Process the 'gt' .cvs file for both training and testing (labels)
            process_file_label(os.path.join(PATH, 'gt_train.csv'), os.path.join(PATH, 'gt_train.csv'), dataset=dataset)

            if benchmark == True:
                # Process the 'gt' folders for both training and testing (masks)
                process_folder_mask(os.path.join(PATH, 'gt_train'), os.path.join(PATH, 'gt_train'), dataset=dataset)

def split_folders(DATA_PATH, p_test=0.2, type='segmentation', metadata=False, seed=None):
    """
    Split the dataset into training and testing folders.

    Args:
        DATA_PATH (str): Path to the main directory containing 'data_train', 'data_test', 'gt_train', and 'gt_test' folders.
        p_test (float): Proportion of samples to be used for testing. Default is 0.2.
        type (str): Type of dataset ('segmentation', 'classification', or 'classification_benchmark'). Default is 'segmentation'.
        metadata (bool): Flag indicating whether metadata is used. Default is False.
        seed (int): Seed for random operations. Default is None.

    Returns:
        None
    """
    # Set the seed for reproducibility
    random.seed(seed)

    data_path_train = os.path.join(DATA_PATH, 'data_train')
    data_path_test = os.path.join(DATA_PATH, 'data_test')
    os.makedirs(data_path_test, exist_ok=True)

    if type == 'segmentation' or type == 'classification_benchmark':
        gt_path_train = os.path.join(DATA_PATH, 'gt_train')
        gt_path_test = os.path.join(DATA_PATH, 'gt_test')
        os.makedirs(gt_path_test, exist_ok=True)

    if type == 'classification' or type == 'classification_benchmark':
        csv_path_train = os.path.join(DATA_PATH, 'gt_train.csv')
        csv_path_test = os.path.join(DATA_PATH, 'gt_test.csv')
        if metadata:
            metadata_path_train = os.path.join(DATA_PATH, 'metadata_train.csv')
            metadata_path_test = os.path.join(DATA_PATH, 'metadata_test.csv')

    # Get a list of files with the '.jpg' extension in the input data folder
    data_files = [f for f in os.listdir(data_path_train) if f.endswith('.jpg')]

    # Calculate the number of samples for testing
    num_test_samples = int(len(data_files) * p_test)

    # Randomly select test samples
    test_samples = random.sample(data_files, num_test_samples)

    # Move testing samples to the test folders
    for test_sample in test_samples:
        data_file_path = os.path.join(data_path_train, test_sample)
        shutil.move(data_file_path, os.path.join(data_path_test, test_sample))

        if type == 'segmentation' or type == 'classification_benchmark':
            mask_file_path = os.path.join(gt_path_train, os.path.splitext(test_sample)[0] + '_segmentation.png')
            shutil.move(mask_file_path, os.path.join(gt_path_test, os.path.splitext(test_sample)[0] + '_segmentation.png'))
        
    if type == 'classification' or type == 'classification_benchmark':
        # Read the training CSV file
        with open(csv_path_train, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            rows = list(csv_reader)

        # Extract the header and the rest of the rows
        header = rows[0]
        data_rows = rows[1:]

        # Remove file extensions from test_samples
        test_samples_without_extension = [os.path.splitext(sample)[0] for sample in test_samples]

        # Filter out rows corresponding to test samples
        test_rows = [row for row in data_rows if os.path.splitext(row[0])[0] in test_samples_without_extension]

        # Write the header and the filtered rows to the test CSV file
        with open(csv_path_test, 'w', newline='') as csv_test_file:
            csv_test_writer = csv.writer(csv_test_file)
            csv_test_writer.writerow(header)
            csv_test_writer.writerows(test_rows)

            # Optional: If you want to remove the test samples from the training CSV file
            remaining_rows = [row for row in data_rows if os.path.splitext(row[0])[0] not in test_samples_without_extension]
            with open(csv_path_train, 'w', newline='') as csv_train_file:
                csv_train_writer = csv.writer(csv_train_file)
                csv_train_writer.writerow(header)
                csv_train_writer.writerows(remaining_rows)
        
        if metadata:
            # Read the training CSV file
            with open(metadata_path_train, 'r') as csv_file:
                csv_reader = csv.reader(csv_file)
                rows = list(csv_reader)

            # Extract the header and the rest of the rows
            header = rows[0]
            data_rows = rows[1:]

            # Filter out rows corresponding to test samples
            test_rows = [row for row in data_rows if os.path.splitext(row[0])[0] in test_samples_without_extension]

            # Write the header and the filtered rows to the test CSV file
            with open(metadata_path_test, 'w', newline='') as csv_test_file:
                csv_test_writer = csv.writer(csv_test_file)
                csv_test_writer.writerow(header)
                csv_test_writer.writerows(test_rows)

                # Optional: If you want to remove the test samples from the training CSV file
                remaining_rows = [row for row in data_rows if os.path.splitext(row[0])[0] not in test_samples_without_extension]
                with open(metadata_path_train, 'w', newline='') as csv_train_file:
                    csv_train_writer = csv.writer(csv_train_file)
                    csv_train_writer.writerow(header)
                    csv_train_writer.writerows(remaining_rows)

'''
---------------------------------------------------------------------------------------------
Model Evaluation
---------------------------------------------------------------------------------------------
'''
class DiceLoss(nn.Module):
    """
    Dice Loss for semantic segmentation.

    Attributes:
        None
    """
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, predicted, target):
        """
        Calculate the Dice coefficient loss for each sample in the batch.

        Args:
            predicted (torch.Tensor): Predicted segmentation maps.
            target (torch.Tensor): Ground truth segmentation maps.

        Returns:
            torch.Tensor: Complement of the average Dice coefficient loss over the batch.
        """
        batch = predicted.size()[0]
        batch_loss = 0
        for index in range(batch):
            coefficient = self._dice_coefficient(
                predicted[index], target[index])
            batch_loss += coefficient

        # Average the Dice coefficient loss over the batch
        batch_loss = batch_loss / batch

        # Return the complement of the average Dice coefficient loss
        return 1 - batch_loss

    def _dice_coefficient(self, predicted, target):
        """
        Calculate the Sørensen–Dice coefficient for a single sample.

        Args:
            predicted (torch.Tensor): Predicted segmentation map for a single sample.
            target (torch.Tensor): Ground truth segmentation map for a single sample.

        Returns:
            float: Sørensen–Dice coefficient for the input sample.
        """
        smooth = 1
        product = torch.mul(predicted, target)
        intersection = product.sum()
        coefficient = (2 * intersection + smooth) / (predicted.sum() + target.sum() + smooth)
        return coefficient
    
class BCEDiceLoss(nn.Module):
    """ 
    Combination of Binary Cross Entropy Loss and Soft Dice Loss.
    """

    def __init__(self, device):
        super(BCEDiceLoss, self).__init__()
        # Instantiate the DiceLoss and move it to the specified device
        self.dice_loss = DiceLoss().to(device)

    def forward(self, predicted, target):
        # Calculate the combined loss by summing Binary Cross Entropy and Dice Loss
        return Fu.binary_cross_entropy(predicted, target) + self.dice_loss(predicted, target)

def _dice_coefficient(output, mask):
    """
    Calculates the Sørensen–Dice Coefficient for a single sample.

    Args:
        output (torch.Tensor): The prediction from the model.
        mask (torch.Tensor): The ground truth.

    Returns:
        float: Sørensen–Dice Coefficient for the input sample.
    """
    smooth = 1
    product = torch.mul(output, mask)
    intersection = torch.sum(product)
    coefficient = (2 * intersection + smooth) / (torch.sum(output) + torch.sum(mask) + smooth)
    
    return coefficient.item()

def pixel_wise_accuracy(output, mask):
    """
    Calculates the pixel-wise accuracy for a single sample.

    output: the prediction from the model.
    mask: the ground truth.
    """
    correct_pixels = torch.sum(output == mask).item()
    total_pixels = mask.numel()
    accuracy = correct_pixels / total_pixels
    
    return accuracy

def get_scores_SEG(trainer, dataset_test, dataset_train=None, processing=False, threshold=0.5 , verbose=True):
    """
    Calculate and print average Dice Coefficient and Pixel-wise Accuracy for training and test datasets.

    Args:
    - trainer: The trainer for the segmentation model.
    - dataset_test: The test dataset.
    - dataset_train (optional): The training dataset.
    - processing (bool): Flag to apply post-processing.
    - threshold: The prediction threshold.
    - verbose (bool): Flag to print the results.

    Returns:
    - Tuple: If verbose is False, returns a tuple of average Dice Coefficient and Pixel-wise Accuracy for training and test datasets.
    """

    # Initialize variables to store cumulative metrics
    dice_coef_train = 0
    dice_coef_test = 0
    accuracy_train = 0
    accuracy_test = 0

    if dataset_train is not None:
        # Calculate metrics for the training dataset
        for i in range(len(dataset_train)):  # Exclude the last element
            _, mask, output = trainer.predict(dataset_train[i], threshold=threshold)

            # Apply post-processing if specified
            if processing == True:
                output = dataset.keep_main_region(output[0])

            # Update cumulative metrics
            dice_coef_train += _dice_coefficient(output, mask)
            accuracy_train += pixel_wise_accuracy(output, mask)

        # Calculate average metrics for the training dataset
        av_dice_coef_train = dice_coef_train / len(dataset_train)
        av_accuracy_train = accuracy_train / len(dataset_train)

    # Calculate metrics for the test dataset
    for i in range(len(dataset_test)):  # Exclude the last element
        _, mask, output = trainer.predict(dataset_test[i], threshold=threshold)

        # Apply post-processing if specified
        if processing == True:
            output = dataset.keep_main_region(output[0])

        # Update cumulative metrics
        dice_coef_test += _dice_coefficient(output, mask)
        accuracy_test += pixel_wise_accuracy(output, mask)

    # Calculate average metrics for the test dataset
    av_dice_coef_test = dice_coef_test / len(dataset_test)
    av_accuracy_test = accuracy_test / len(dataset_test)
    
    if verbose:
        if dataset_train is not None:
            # Print and return the results
            # Define the metric names
            metric_names = ['Pixel-wise Accuracy', 'Dice Coefficient']

            # Define the metric values
            train_metrics = [av_accuracy_train, av_dice_coef_train]
            test_metrics = [av_accuracy_test, av_dice_coef_test]

            # Print the metrics in a table-like structure
            print("{:<30} {:<15} {:<15}".format("Metric", "Train", "Test"))
            print("="*60)

            for name, train, test in zip(metric_names, train_metrics, test_metrics):
                print("{:<30} {:<15.4f} {:<15.4f}".format(name, train, test))
        else:
            # Define the metric names
            metric_names_test = ['Pixel-wise Accuracy', 'Dice Coefficient']

            # Define the metric values for the test set
            test_metrics = [av_accuracy_test, av_dice_coef_test]

            # Print the metrics for the test set in a table-like structure
            print("{:<30} {:<15}".format("Metric", "Test"))
            print("="*45)

            for name, test in zip(metric_names_test, test_metrics):
                print("{:<30} {:<15.4f}".format(name, test))

    if verbose == False:
        if dataset_train is not None: 
            return av_dice_coef_train, av_dice_coef_test, av_accuracy_train, av_accuracy_test
        else:
            return av_dice_coef_test, av_accuracy_test

def accuracy_CLASS(output, label):
    """
    Calculate accuracy for a binary classification task.

    Parameters:
    - output: Predicted label (0 or 1)
    - label: Ground truth label (0 or 1)

    Returns:
    - accuracy: Accuracy of the prediction
    """

    # Ensure that the predicted labels are either 0 or 1
    if output not in {0, 1}:
        raise ValueError("Labels must be either 0 or 1.")

    # Calculate accuracy
    accuracy = 1 if output == label else 0

    return accuracy

def metrics_CLASS(dataset, trainer, threshold=0.5, metadata=False):
    """
    Calculate specificity, sensitivity, and precision for a given dataset and trainer.

    Args:
        dataset: The dataset for which to calculate specificity, sensitivity, and precision.
        trainer: The trainer for the segmentation model.
        threshold (float): Threshold for binary classification. Default is 0.5.
        metadata (bool): Flag indicating whether metadata is used. Default is False.

    Returns:
        Tuple: Specificity, sensitivity, and precision.
    """
    true_negatives = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for i in range(len(dataset)):
        _, label, output = trainer.predict(dataset[i], threshold=threshold, task='classification', metadata=metadata)

        # Update confusion matrix values
        true_negatives += (output == 0) and (label == 0)
        true_positives += (output == 1) and (label == 1)
        false_positives += (output == 1) and (label == 0)
        false_negatives += (output == 0) and (label == 1)

    # Calculate specificity and sensitivity
    precision   = true_positives / (true_positives + false_positives) if (true_positives + false_positives) != 0 else 0
    specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) != 0 else 0
    sensitivity = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) != 0 else 0

    return precision, specificity, sensitivity

def get_scores_CLASS(trainer, dataset_test, dataset_train=None, verbose=True, threshold=0.5, metadata=False):
    """
    Calculate and print average accuracy, specificity, and sensitivity for training and test datasets.

    Args:
        model: The segmentation model.
        trainer: The trainer for the segmentation model.
        dataset_test: The test dataset.
        dataset_train (optional): The training dataset.
        verbose (bool): Flag to print the results.
        threshold (float): Threshold for binary classification. Default is 0.5.
        metadata (bool): Flag indicating whether metadata is used. Default is False.

    Returns:
        Tuple: If verbose is False, returns a tuple of average accuracy, specificity, and sensitivity for training and test datasets.
    """
    # Initialize variables to store cumulative metrics
    accuracy_train = 0
    specificity_train = 0
    sensitivity_train = 0

    accuracy_test = 0
    specificity_test = 0
    sensitivity_test = 0

    if dataset_train is not None:
        j = 0
        # Calculate metrics for the training dataset
        for i in range(len(dataset_train)):
            if metadata == False:
                _, label, output = trainer.predict(dataset_train[i], threshold=threshold, task='classification', metadata=metadata)

            if output != 0:
              j = j + 1

            # Update cumulative metrics
            accuracy_train += accuracy_CLASS(output, label)
        
        print(f'Number of positive pred. TRAIN: {j}')

        # Calculate average accuracy for the training dataset
        accuracy_train /= len(dataset_train)

        # Calculate specificity and sensitivity for the training dataset
        precision_train, specificity_train, sensitivity_train = metrics_CLASS(dataset_train, trainer, threshold=threshold, metadata=metadata)

        # Calculate the F1 score
        f1_train = 2 * (precision_train * sensitivity_train) / (precision_train + sensitivity_train) if (precision_train + sensitivity_train) != 0 else 0

    j = 0
    # Calculate metrics for the test dataset
    for i in range(len(dataset_test)):
        _, label, output = trainer.predict(dataset_test[i], threshold=threshold, task='classification', metadata=metadata)

        if output != 0:
          j = j + 1

        # Update cumulative metrics
        accuracy_test += accuracy_CLASS(output, label)
    print(f'Number of positive pred. TEST: {j}')
    print('')

    # Calculate average accuracy for the test dataset
    accuracy_test /= len(dataset_test)

    # Calculate specificity and sensitivity for the test dataset
    precision_test, specificity_test, sensitivity_test = metrics_CLASS(dataset_test, trainer, threshold=threshold, metadata=metadata)

    # Calculate the F1 score
    f1_test = 2 * (precision_test * sensitivity_test) / (precision_test + sensitivity_test) if (precision_test + sensitivity_test) != 0 else 0

    if verbose:
        if dataset_train is not None:
            # Print and return the results
            # Define the metric names
            metric_names = ['Accuracy', 'Specificity', 'Sensitivity', 'F1 Score']

            # Define the metric values
            train_metrics = [accuracy_train, specificity_train, sensitivity_train, f1_train]
            test_metrics = [accuracy_test, specificity_test, sensitivity_test, f1_test]

            # Print the metrics in a table-like structure
            print("{:<30} {:<15} {:<15}".format("Metric", "Train", "Test"))
            print("="*60)

            for name, train, test in zip(metric_names, train_metrics, test_metrics):
                print("{:<30} {:<15.4f} {:<15.4f}".format(name, train, test))
        else:
            # Define the metric names
            metric_names_test = ['Accuracy', 'Specificity', 'Sensitivity', 'F1 Score']

            # Define the metric values for the test set
            test_metrics = [accuracy_test, specificity_test, sensitivity_test, f1_test]

            # Print the metrics for the test set in a table-like structure
            print("{:<30} {:<15}".format("Metric", "Test"))
            print("="*45)

            for name, test in zip(metric_names_test, test_metrics):
                print("{:<30} {:<15.4f}".format(name, test))

    if not verbose:
        if dataset_train is not None:
            return accuracy_train, specificity_train, sensitivity_train, accuracy_test, specificity_test, sensitivity_test
        else:
            return accuracy_test, specificity_test, sensitivity_test
        
'''
---------------------------------------------------------------------------------------------
Other Utils
---------------------------------------------------------------------------------------------
'''
def delete_folder(folder_path):
    """
    Delete the specified folder and its contents.

    Args:
        folder_path (str): Path to the folder to be deleted.
    """
    try:
        # Check if the folder exists
        if os.path.exists(folder_path):
            # Delete the folder and its contents
            shutil.rmtree(folder_path)
    except Exception as e:
        print(f"Error deleting folder '{folder_path}': {e}")

def get_label_index(csv_path, labelOI=1):
    """
    Retrieve the indices associated with a specific label from a CSV file.

    Args:
        csv_path (str): Path to the CSV file containing index-label pairs.
        labelOI (float, optional): The label of interest. Default is 1.

    Returns:
        list: List of indices with the specified label.
        int: The last index found in the CSV file.
    """
    zero_labels = []
    last_index = 0

    # Open the CSV file in read mode
    with open(csv_path, 'r') as file:
        # Iterate through each line in the file
        for line in file:
            # Assuming the delimiter is '\t', update if necessary
            columns = line.strip().split('\t')

            # Assuming the first column is the index column
            index_value = int(columns[0])
            label = float(columns[1])

            # Check if the label matches the specified labelOI
            if label == labelOI:
                zero_labels.append(index_value)

            # Update the last index
            last_index = max(last_index, index_value)

    return zero_labels, last_index

def numerical_sort(value):
    """
    Extract the numeric part of the file name using a regular expression for numerical sorting.

    Args:
        value (str): The input string to extract numeric values from.

    Returns:
        int: Extracted numeric value.
    """
    # Extracts the numeric part of the file name using a regular expression
    numbers = re.findall(r'\d+', value)
    return int(numbers[0]) if numbers else float('inf')

def get_labels(lesion_dataset):
    """
    Retrieve the labels from a lesion dataset.

    Args:
        lesion_dataset (Dataset): The lesion dataset.

    Returns:
        np.array: Array of labels.
    """
    labels = []
    
    for i in range(len(lesion_dataset)):
        data = lesion_dataset[i]
        labels.append(data['label'])
    
    return np.array(labels)