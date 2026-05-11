import os
import random
import cv2

import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision import transforms as T

import numpy as np
import pandas as pd

from skimage.draw import draw
from PIL import Image


class LesionDataset(Dataset):
    def __init__(self, root_dir, trainer_seg=None, TYPE='train', task='segmentation', metadata=False):
        """
        Initialize the LesionDataset.

        Args:
            root_dir (str): Root directory containing image and mask data.
            TYPE (str, optional): Dataset type (e.g., 'train', 'test'). Default is 'train'.
            task (str, optional): Task type, either 'segmentation' or 'classification'. Default is 'segmentation'.
            trainer_seg (Trainer, optional): Trainer object for segmentation task. Default is None.
            metadata (bool): If True, extract metadata from the dataset. Default is False.
        """
        self.root_dir = root_dir
        self.TYPE = TYPE
        self.task = task
        self.trainer_seg = trainer_seg
        self.metadata = metadata

        # Resize images to 256x256
        self.resize256 = transforms.Resize(size=(256, 256))

        if metadata == True:
            # Extract metadata if provided
            metadata = extract_metadata(root_dir)
            
            # Extract metadata lists
            self.age_approx_list = metadata[0]
            self.anatom_site_general_list = metadata[1]
            self.sex_list = metadata[2]
            # Combine the three lists into a single tensor
            self.metadata_tensor = torch.tensor([self.age_approx_list, self.anatom_site_general_list, self.sex_list], dtype=torch.float32).t()

    def transform(self, image, mask=None):
        """
        Applies data transformations to the input image and mask.

        Args:
            image (PIL.Image): The input image to be transformed.
            mask (PIL.Image, optional): The corresponding mask for the image. Default is None.

        Returns:
            torch.Tensor: Transformed image represented as a PyTorch tensor.
            torch.Tensor: Transformed mask represented as a PyTorch tensor (if mask is not None).
                        If mask is None, returns None for the mask.
        """ 
        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            if mask is not None:
                mask = TF.hflip(mask)

        # Random vertical flipping
        if random.random() > 0.5:
            image = TF.vflip(image)
            if mask is not None:
                mask = TF.vflip(mask)

        # Random rotation up to 45 degrees
        if random.random() > 0.5:
            angle = random.uniform(-45, 45)
            image = T.RandomRotation(degrees=(angle, angle))(image)
            if mask is not None:
                mask = T.RandomRotation(degrees=(angle, angle))(mask)

        # Random zoom with a maximum of 120%
        if random.random() > 0.5:
            scale = random.uniform(1.0, 1.2)  # 1.0 corresponds to no zoom, 1.2 corresponds to 120% zoom
            image = T.RandomAffine(degrees=0, translate=(0, 0), scale=(scale, scale))(image)
            if mask is not None:
                mask = T.RandomAffine(degrees=0, translate=(0, 0), scale=(scale, scale))(mask)
        
        # Transformation applied excusively to the classification task
        if self.task == 'classification' or self.task == 'classification_benchmark':            
            # Translation of the image to tensor
            image = TF.to_tensor(image)
            
            # Resize transformation to match the expected input size
            image = self.resize256(image)

            # Add artificial hair
            if random.random() > 0.5:
                image = add_hair(image, num_hairs=random.randint(10, 40), hair_length=random.randint(300, 500), num_segments=random.randint(3, 6), hair_color=random.uniform(0., 0.2))    

            if self.task == 'classification':
                # Extract the region of interest from the provided segmentation model
                image = roi(image, self.trainer_seg)
            
            elif self.task == 'classification_benchmark':
                # Translation of mask to tensor
                mask = TF.to_tensor(mask)
                mask = self.resize256(mask)
                # Extract the region of interest from the groundtruth
                image = roi_benchmark(image, mask)
            
            # Apply color jittering with a specified probability
            if random.random() > 0.5:
                # Define the range of jittering parameters (e.g., brightness, contrast, saturation, hue)
                color_jitter = transforms.ColorJitter(
                    brightness= 0.25,
                    contrast=0.25,
                    saturation=0.1,
                    hue=0.1
                )
                # Apply color jittering to the image
                image = color_jitter(image)

            # Random cutouts
            cutout = T.RandomErasing(p=0.25, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0, inplace=False)
            image = cutout(image)
                
        # Transformation for segmentation only
        if self.task == 'segmentation':
            # Convert image and mask to tensor
            image = TF.to_tensor(image)
            mask = TF.to_tensor(mask)

            # Resize image and mask to 256x256
            image = self.resize256(image)
            mask = self.resize256(mask)

            # Add artificial hair
            if random.random() > 0.25:
                image = add_hair(image, num_hairs=random.randint(10, 40), hair_length=random.randint(300, 500), num_segments=random.randint(3, 6), hair_color=random.uniform(0., 0.2))    

            # Cutmix transformation
            if random.random() > 0.75:
                random_index = random.randint(0, len(self) - 1)
                random_image_name = os.path.join(self.root_dir, f'data_{self.TYPE}', f'{random_index}.png')
                random_mask_name = os.path.join(self.root_dir, f'gt_{self.TYPE}', f'{random_index}.png')
            
                random_image = Image.open(random_image_name)
                random_mask = Image.open(random_mask_name)
                random_mask = random_mask.convert('L')

                random_image = TF.to_tensor(random_image)
                random_mask = TF.to_tensor(random_mask)

                image, mask = cutmix(image, mask, random_image, random_mask)
                
            # Mosaic transformation
            elif random.random() > 0.75:
                i = random.randint(0, len(self) - 1)
                image1 = Image.open(os.path.join(self.root_dir, f'data_{self.TYPE}', f'{i}.png'))
                mask1 = Image.open(os.path.join(self.root_dir, f'gt_{self.TYPE}', f'{i}.png'))
                mask1 = mask1.convert('L')
                image1 = TF.to_tensor(image1)
                mask1 = TF.to_tensor(mask1)
                
                i = random.randint(0, len(self) - 1)
                image2 = Image.open(os.path.join(self.root_dir, f'data_{self.TYPE}', f'{i}.png'))
                mask2 = Image.open(os.path.join(self.root_dir, f'gt_{self.TYPE}', f'{i}.png'))
                mask2 = mask2.convert('L')
                image2 = TF.to_tensor(image2)
                mask2 = TF.to_tensor(mask2)
            
                i = random.randint(0, len(self) - 1)
                image3 = Image.open(os.path.join(self.root_dir, f'data_{self.TYPE}', f'{i}.png'))
                mask3 = Image.open(os.path.join(self.root_dir, f'gt_{self.TYPE}', f'{i}.png'))
                mask3 = mask3.convert('L')
                image3 = TF.to_tensor(image3)
                mask3 = TF.to_tensor(mask3)
                
                image, mask = mosaic(image, mask, image1, mask1, image2, mask2, image3, mask3)

        # Return image and mask if we have mask (segmentation)
        if mask is not None:
            return image, mask
        # Else return only image
        else:
            return image

    def __getitem__(self, index):
        """
        Get a sample from the dataset by index.

        Args:
            index (int): Index of the sample.

        Returns:
            sample (dict): A dictionary containing the index, transformed image, and transformed mask.
        """
        # Load the image using its index
        image_name = os.path.join(self.root_dir, f'data_{self.TYPE}', f'{index}.png')
        image = Image.open(image_name)

        # Load the corresponding mask for segmentation task
        if self.task == 'segmentation' or self.task == 'classification_benchmark': 
            mask_name = os.path.join(self.root_dir, f'gt_{self.TYPE}', f'{index}.png')
            mask = Image.open(mask_name)
            mask = mask.convert('L')
        
        # Load label from CSV file for classification task
        if self.task == 'classification' or self.task == 'classification_benchmark':
            label_file = os.path.join(self.root_dir, f'gt_{self.TYPE}.csv')
            label = get_label_from_index(label_file, index)

        # Apply data transformation for training set
        if self.TYPE == 'train':
            # Apply transformations for training data
            if self.task == 'segmentation': 
                image, mask = self.transform(image, mask=mask)
            elif self.task == 'classification': 
                image = self.transform(image)
            elif self.task == 'classification_benchmark': 
                image, mask = self.transform(image, mask=mask)
        
        # Apply data transformation for testing set
        elif self.TYPE == 'test':
            if self.task == 'segmentation': 
                image = TF.to_tensor(image)
                mask = TF.to_tensor(mask)
                image = self.resize256(image)
                mask = self.resize256(mask)
            elif self.task == 'classification': 
                image = TF.to_tensor(image)
                image = self.resize256(image)
                image = roi(image, self.trainer_seg)
            elif self.task == 'classification_benchmark':
                image = TF.to_tensor(image)
                mask = TF.to_tensor(mask) 
                image = self.resize256(image)
                mask = self.resize256(mask)
                image = roi_benchmark(image, mask)
        
        # Prepare the sample dictionary based on the task
        if self.task == 'segmentation':
            sample = {
                'index': int(index),
                'image': image,
                'mask': mask
            }
        elif self.task == 'classification':
            if self.metadata == True:
                sample = {
                    'index': int(index),
                    'image': image,
                    'metadata': self.metadata_tensor[index],
                    'label': int(float(label))
                }
            else:
                sample = {
                    'index': int(index),
                    'image': image,
                    'label': int(float(label))
                }
        elif self.task == 'classification_benchmark':
            if self.metadata == True:
                sample = {
                    'index': int(index),
                    'image': image,
                    'metadata': self.metadata_tensor[index],
                    'label': int(float(label)),
                    'mask': mask
                }
            else:
                sample = {
                    'index': int(index),
                    'image': image,
                    'label': int(float(label)),
                    'mask': mask
                }

        # Return the sample
        return sample

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            size_of_dataset (int): Number of samples in the dataset.
        """
        # Get the list of image files in the corresponding data folder
        image_files = [file for file in os.listdir(os.path.join(self.root_dir, f'data_{self.TYPE}')) if file.endswith('.png')]

        # Calculate and return the number of samples in the dataset
        size_of_dataset = len(image_files)

        return size_of_dataset
    
    def get_labels(self):
        """
        Retrieves labels from the ground truth CSV file associated with the dataset.

        Returns:
            torch.Tensor: Tensor containing the labels, represented as integers, with dtype=torch.long.
        """
        # Construct the path to the ground truth CSV file
        label_file = os.path.join(self.root_dir, f'gt_{self.TYPE}.csv')

        # Initialize an empty list to store labels
        labels = []

        # Iterate through each index in the dataset
        for index in range(len(self)):
            # Retrieve the label for the current index using the helper function
            labels.append(get_label_from_index(label_file, index))

        # Convert the labels to integers
        labels = [int(label) for label in labels]

        # Convert the list of labels to a PyTorch tensor with long datatype
        return torch.tensor(labels, dtype=torch.long)

'''
---------------------------------------------------------------------------------------------
Data Transformation Functions
---------------------------------------------------------------------------------------------
'''
def add_hair(image, num_hairs=20, hair_length=400, num_segments=6, hair_color=0.5, image_size=256):
    """
    Add artificial undulated hairs to a given skin lesion image.

    Parameters:
    - image (torch.Tensor): input image with shape (3, 256, 256)
    - num_hairs (int): number of hairs to add
    - hair_length (int): length of the added hair
    - num_segments (int): number of segments in each hair
    - hair_color (float): grayscale value of the added hair (0 to 1)
    - image_size (int): the original image size

    Returns:
    - (torch.Tensor) image with added hair
    """
    # Create a copy of the original image
    image_with_hair = image.clone()

    for _ in range(num_hairs):
        # Generate random hair parameters
        start_x = np.random.randint(0, image_size)
        start_y = np.random.randint(0, image_size)

        # Generate random angles for each segment
        angles = np.random.uniform(0, 1/3 * np.pi, num_segments)

        # Draw the undulated hair on the image
        for i in range(num_segments - 1):
            segment_length = hair_length / num_segments
            end_x = start_x + int(segment_length * np.cos(angles[i]))
            end_y = start_y + int(segment_length * np.sin(angles[i]))

            rr, cc = draw.line(start_y, start_x, end_y, end_x)
            rr = np.clip(rr, 0, image_size -1)
            cc = np.clip(cc, 0, image_size -1)
            image_with_hair[:, rr, cc] = hair_color

            start_x, start_y = end_x, end_y

    return image_with_hair

def cutmix(image1, mask1, image2, mask2):
    """
    Apply CutMix data augmentation.

    Args:
        image1 (torch.Tensor): First image (3xHxW).
        mask1 (torch.Tensor): Corresponding mask for the first image (1xHxW).
        image2 (torch.Tensor): Second image (3xHxW).
        mask2 (torch.Tensor): Corresponding mask for the second image (1xHxW).

    Returns:
        image (torch.Tensor): Transformed image.
        mask (torch.Tensor): Transformed mask.
    """
    beta = np.random.uniform(0.3, 0.7)  # CutMix ratio parameter

    _, h, w = image1.size()
    cut_h = int(h * beta)
    cut_w = int(w * beta)

    # Generate random coordinates for CutMix
    y1 = np.random.randint(0, h - cut_h)
    y2 = y1 + cut_h
    x1 = np.random.randint(0, w - cut_w)
    x2 = x1 + cut_w

    # Randomly select a corner for fixing image2
    fixed_corner = np.random.choice(["top-left", "top-right", "bottom-left", "bottom-right"])

    # Determine the fixed corner coordinates
    if fixed_corner == "top-left":
        fixed_x = 0
        fixed_y = 0
    elif fixed_corner == "top-right":
        fixed_x = w - cut_w
        fixed_y = 0
    elif fixed_corner == "bottom-left":
        fixed_x = 0
        fixed_y = h - cut_h
    elif fixed_corner == "bottom-right":
        fixed_x = w - cut_w
        fixed_y = h - cut_h

    # Apply CutMix
    mixed_image = image1.clone()
    mixed_mask = mask1.clone()

    if fixed_corner != "none":
        mixed_image[:, fixed_y:fixed_y + cut_h, fixed_x:fixed_x + cut_w] = image2[:, y1:y2, x1:x2]
        mixed_mask[:, fixed_y:fixed_y + cut_h, fixed_x:fixed_x + cut_w] = mask2[:, y1:y2, x1:x2]
    else:
        mixed_image[:, y1:y2, x1:x2] = image2[:, y1:y2, x1:x2]
        mixed_mask[:, y1:y2, x1:x2] = mask2[:, y1:y2, x1:x2]

    return mixed_image, mixed_mask

def mosaic(image1, mask1, image2, mask2, image3, mask3, image4, mask4, image_size=256):
    """
    Combine four images and masks by placing each resized image in one corner.

    Args:
        image1 (torch.Tensor): First image (3xHxW).
        mask1 (torch.Tensor): Corresponding mask for the first image (1xHxW).
        image2 (torch.Tensor): Second image (3xHxW).
        mask2 (torch.Tensor): Corresponding mask for the second image (1xHxW).
        image3 (torch.Tensor): Third image (3xHxW).
        mask3 (torch.Tensor): Corresponding mask for the third image (1xHxW).
        image4 (torch.Tensor): Fourth image (3xHxW).
        mask4 (torch.Tensor): Corresponding mask for the fourth image (1xHxW).

    Returns:
        combined_image (torch.Tensor): Combined image.
        combined_mask (torch.Tensor): Combined mask.
    """
    # Resize each image and mask to the target size
    resize = transforms.Resize(size=(int(image_size/2), int(image_size/2)))
    image1_resized = resize(image1)
    image2_resized = resize(image2)
    image3_resized = resize(image3)
    image4_resized = resize(image4)

    mask1_resized = resize(mask1)
    mask2_resized = resize(mask2)
    mask3_resized = resize(mask3)
    mask4_resized = resize(mask4)

    # Create a blank canvas for the combined image
    combined_image = torch.zeros_like(image1)
    combined_mask = torch.zeros_like(mask1)

    _, h, w = image1.size()

    # Place each resized image in one corner of the combined image
    combined_image[:, :h//2, :w//2] = image1_resized
    combined_image[:, :h//2, w//2:] = image2_resized
    combined_image[:, h//2:, :w//2] = image3_resized
    combined_image[:, h//2:, w//2:] = image4_resized

    # Combine resized masks similarly
    combined_mask[:, :h//2, :w//2] = mask1_resized
    combined_mask[:, :h//2, w//2:] = mask2_resized
    combined_mask[:, h//2:, :w//2] = mask3_resized
    combined_mask[:, h//2:, w//2:] = mask4_resized

    return combined_image, combined_mask

'''
---------------------------------------------------------------------------------------------
Post-Processing Functions
---------------------------------------------------------------------------------------------
'''
def keep_main_region(mask_tensor):
    """
    Extracts the main region from a binary mask using contours and converts it back to a PyTorch tensor.

    Parameters:
    - mask_tensor (torch.Tensor): The binary mask represented as a PyTorch tensor.

    Returns:
    - torch.Tensor: The PyTorch tensor representing the main region extracted from the input mask.
    """

    # Convert the PyTorch tensor to a NumPy array
    mask_np = mask_tensor.cpu().numpy()

    # Convert the mask to uint8 and to a single-channel image
    mask_uint8 = (mask_np[0] * 255).astype(np.uint8)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest area
    if contours:
        main_contour = max(contours, key=cv2.contourArea)

        # Create an empty mask
        main_mask = np.zeros_like(mask_uint8)

        # Draw the main contour on the empty mask
        cv2.drawContours(main_mask, [main_contour], -1, 1, thickness=cv2.FILLED)

        # Apply a threshold to ensure values are 0 or 1
        main_mask = np.where(main_mask > 0.5, 1, 0)

        # Convert the mask back to a PyTorch tensor
        main_mask_tensor = torch.unsqueeze(torch.from_numpy(main_mask.astype(np.float32)), 0)

        return main_mask_tensor
    else:
        # If no contours found, return the original mask as PyTorch tensor
        return mask_tensor

def roi(image, trainer):
    """
    Extracts the Region of Interest (ROI) from an input image using a segmentation model.

    Parameters:
    - image (torch.Tensor): The input image for which the ROI needs to be extracted.
    - trainer: The segmentation model trainer responsible for predicting the ROI.

    Returns:
    - torch.Tensor: The image with the ROI retained, while other the backgrounf is masked.
    """
    # Prepare data for prediction, setting initial values for index and mask
    data = {'index': '_', 'image': image, 'mask': '_'}
  
    # Use the trainer to predict segmentation output
    _, _, output = trainer.predict(data, task='segmentation')

    # Keep the main region from the segmentation output
    processed_output = keep_main_region(output[0])

    # Check if there are any non-zero elements
    if torch.nonzero(processed_output[0]).size(0) != 0:
      # Find bounding box coordinates
      nonzero_coords = torch.nonzero(processed_output[0])
      min_coords = torch.min(nonzero_coords, dim=0).values
      max_coords = torch.max(nonzero_coords, dim=0).values

      #Crop the image
      processed_output = processed_output[:, min_coords[0]:max_coords[0]+1, min_coords[1]:max_coords[1]+1]
      image = image[:, min_coords[0]:max_coords[0]+1, min_coords[1]:max_coords[1]+1]

      # Ensure the cropped image is squared
      max_size = max(processed_output.size(1), processed_output.size(2))
      processed_output = F.pad(processed_output, (0, max_size - processed_output.size(2), 0, max_size - processed_output.size(1)))
      image = F.pad(image, (0, max_size - image.size(2), 0, max_size - image.size(1)))

    # Resize to 256x256
    resize256 = transforms.Resize(size=(256, 256))
    image = resize256(image)
    processed_output = resize256(processed_output)    
    
    # Multiply the input image with the expanded segmentation output to retain the ROI
    roi_image = torch.mul(image, processed_output)
    
    return roi_image

def roi_benchmark(image, mask):
    """
    Extracts the Region of Interest (ROI) from an input image using the grountruth mask.

    Parameters:
    - image (torch.Tensor): The input image for which the ROI needs to be extracted.
    - mask (torch.Tensor): The corresponding grountruth mask.

    Returns:
    - torch.Tensor: The image with the ROI retained, while other the backgrounf is masked.
    """  
    output = mask.clone()
    processed_output = output

    # Check if there are any non-zero elements
    if torch.nonzero(processed_output[0]).size(0) != 0:
      # Find bounding box coordinates
      nonzero_coords = torch.nonzero(processed_output[0])
      min_coords = torch.min(nonzero_coords, dim=0).values
      max_coords = torch.max(nonzero_coords, dim=0).values

      #Crop the image
      processed_output = processed_output[:, min_coords[0]:max_coords[0]+1, min_coords[1]:max_coords[1]+1]
      image = image[:, min_coords[0]:max_coords[0]+1, min_coords[1]:max_coords[1]+1]

      # Ensure the cropped image is squared
      max_size = max(processed_output.size(1), processed_output.size(2))
      processed_output = F.pad(processed_output, (0, max_size - processed_output.size(2), 0, max_size - processed_output.size(1)))
      image = F.pad(image, (0, max_size - image.size(2), 0, max_size - image.size(1)))

    # Resize to 256x256
    resize256 = transforms.Resize(size=(256, 256))
    image = resize256(image)
    processed_output = resize256(processed_output)

    # Multiply the input image with the expanded segmentation output to retain the ROI
    roi_image = torch.mul(image, processed_output)
      
    return roi_image

'''
---------------------------------------------------------------------------------------------
Data Helpers
---------------------------------------------------------------------------------------------
'''
def get_label_from_index(label_file, index):
    """
    Retrieves the label associated with a given index from a label file.

    Parameters:
    - label_file (str): The path to the label file containing index-label pairs.
    - index (int): The index for which the corresponding label needs to be retrieved.

    Returns:
    - str: The label associated with the specified index.
    """

    # Open the label file in read mode
    with open(label_file, 'r') as file:
        # Iterate through each line in the file
        for line in file:
            # Assuming the delimiter is '\t', update if necessary
            columns = line.strip().split('\t')
            
            # Assuming the first column is the index column
            index_value = int(columns[0])

            # Check if the current line's index matches the specified index
            if index_value == index:
                # Retrieve the label associated with the index
                label = columns[1]

    # Return the label associated with the specified index
    return label


'''
---------------------------------------------------------------------------------------------
Metadata Integration
---------------------------------------------------------------------------------------------
'''    
def map_metadata(metadata_list):
    """
    Map unique elements in the input list to normalized values.

    Args:
        metadata_list (list): A list containing metadata elements.

    Returns:
        list: Normalized values corresponding to the input list elements.
    """
    # Create a dictionary to store the mapping
    value_mapping = {}

    # Assign unique values starting from 1
    assigned_values = []

    value_mapping[np.nan] = 0
    for element in metadata_list:
        # Map empty elements to 0
        if element not in value_mapping:
            value_mapping[element] = len(value_mapping) + 1  # Start at 1
        assigned_values.append(value_mapping[element])
    
    # Normalize the assigned values
    max_value = max(assigned_values, default=0)  # Handle case where list is empty
    normalized_values = [value / max_value for value in assigned_values]

    return normalized_values

def extract_metadata(PATH, type='train'):
    """
    Extract metadata information from a CSV file.

    Args:
        PATH (str): Directory containing the CSV file.
        type (str, optional): Type of data ('TRAIN' or 'TEST'). Default is 'TRAIN'.

    Returns:
        tuple: Lists of normalized values for age, anatomical site, and sex.
    """
    # Read the CSV file
    if type == 'train':
        df = pd.read_csv(os.path.join(PATH, 'metadata_train.csv'))
    elif type == 'test':
        df = pd.read_csv(os.path.join(PATH, 'metadata_test.csv'))

    # Initialize lists for each category
    age_approx_list = df['age_approx'].tolist()
    anatom_site_general_list = df['anatom_site_general'].tolist()
    sex_list = df['sex'].tolist()

    # Map metadata values to normalized values
    age_approx_list = map_metadata(age_approx_list)
    anatom_site_general_list = map_metadata(anatom_site_general_list)
    sex_list = map_metadata(sex_list)

    return age_approx_list, anatom_site_general_list, sex_list