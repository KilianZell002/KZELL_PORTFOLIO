import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision.models as models

'''
---------------------------------------------------------------------------------------------
Benchmark
---------------------------------------------------------------------------------------------
'''
class RandomGuessModel(nn.Module):
    def __init__(self, input_channels=3, output_channels=1, task='segmentation'):
        """ 
        A simple PyTorch model that generates random predictions between 0 and 1.

        Parameters:
            input_channels (int): Number of input channels in the input data.
            output_channels (int): Number of output channels in the predicted data.
            task (str): Task type, either 'segmentation' or 'classification'.
        """
        super(RandomGuessModel, self).__init__()

        self.task = task

    def forward(self, x):
        """ 
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Random predictions between 0 and 1 (for segmentation) or a random choice of 0 or 1 (for classification).
        """
        if self.task == 'segmentation':
            # Generate random predictions between 0 and 1 for each pixel
            output = torch.rand(x.size(0), 1, x.size(2), x.size(3))
        
        elif self.task == 'classification':
            # Generate a random choice of 0 or 1 for the label
            output = torch.tensor([random.choice([0, 1])])

        return output

class ConstantModel(nn.Module):
    def __init__(self, input_channels=3, output_channels=1, task='segmentation'):
        """ 
        A simple PyTorch model that generates constant predictions at 0.

        Parameters:
            input_channels (int): Number of input channels in the input data.
            output_channels (int): Number of output channels in the predicted data.
            task (str): Task type, either 'segmentation' or 'classification'.
        """
        super(ConstantModel, self).__init__()

        self.task = task

    def forward(self, x):
        """ 
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Constant predictions of 0 (for segmentation) or 0 (for classification).
        """
        if self.task == 'segmentation':
            # Generate constant predictions of value 0 for each pixel
            output = torch.zeros(x.size(0), 1, x.size(2), x.size(3))
        elif self.task == 'classification':
            # Generate a constant prediction of 0 for the entire batch
            output = torch.tensor([0])

        return output

'''
---------------------------------------------------------------------------------------------
Segmentation
---------------------------------------------------------------------------------------------
'''
class SimpleCNN(nn.Module):
    def __init__(self, input_channels=3, output_channels=1, padding=1, ks=3):
        """ 
        A simple convolutional neural network (CNN) model for binary classification.

        Parameters:
            input_channels (int): Number of input channels in the input data.
            output_channels (int): Number of output channels in the predicted data.
            padding (int): Padding for convolutional layers.
            ks (int): Kernel size for convolutional layers.
        """
        super(SimpleCNN, self).__init__()

        # Convolutional layers with Batch Normalization, ReLU activation, and max pooling
        self.conv1 = nn.Conv2d(input_channels, 16, ks, padding)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, ks, padding)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, ks, padding)
        self.bn3 = nn.BatchNorm2d(64)

        # Adjusting fully connected layers for 256x256 input
        self.fc1 = nn.Linear(240 * 240, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, output_channels * 256 * 256)

        # Resize images to 256x256
        self.resize_transform = transforms.Resize((256, 256))

    def forward(self, x):
        """ 
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after forward pass.
        """
        # Resize image to proper dimension
        x = self.resize_transform(x) 

        # Convolutional layers with Batch Normalization, ReLU activation, and max pooling
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)

        # Flatten the output for fully connected layers
        x = x.view(x.size(0), -1)

        # Fully connected layers with Batch Normalization and ReLU activation
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.fc2(x)

        # Apply sigmoid activation to get binary classification
        output = torch.sigmoid(x)
        output = output.view(-1, 1, 256, 256)

        return output

class UNet(nn.Module):
    """ 
    U-Net architecture for semantic segmentation.
    """

    def __init__(self, filter_num=[8, 16, 32, 64, 128], input_channels=3, output_channels=1, padding=1, ks=3):
        """ 
        Constructor for the UNet class.

        Parameters:
            filter_num (list): A list of the number of filters (number of input or output channels of each layer).
            input_channels (int): Input channels for the network.
            output_channels (int): Output channels for the final network.
            padding (int): Padding for convolutional layers.
            ks (int): Kernel size for convolutional layers.
        """
        # Call the constructor of the parent class
        super(UNet, self).__init__()

        # Encoding part of the network
        # Block 1
        self.conv1_1 = nn.Conv2d(input_channels, filter_num[0], kernel_size=ks, padding=padding)
        self.bn1_1 = nn.BatchNorm2d(filter_num[0])
        self.conv1_2 = nn.Conv2d(filter_num[0], filter_num[0], kernel_size=ks, padding=padding)
        self.bn1_2 = nn.BatchNorm2d(filter_num[0])
        self.maxpool1 = nn.MaxPool2d(2)

        # Block 2
        self.conv2_1 = nn.Conv2d(filter_num[0], filter_num[1], kernel_size=ks, padding=padding)
        self.bn2_1 = nn.BatchNorm2d(filter_num[1])
        self.conv2_2 = nn.Conv2d(filter_num[1], filter_num[1], kernel_size=ks, padding=padding)
        self.bn2_2 = nn.BatchNorm2d(filter_num[1])
        self.maxpool2 = nn.MaxPool2d(2)

        # Block 3
        self.conv3_1 = nn.Conv2d(filter_num[1], filter_num[2], kernel_size=ks, padding=padding)
        self.bn3_1 = nn.BatchNorm2d(filter_num[2])
        self.conv3_2 = nn.Conv2d(filter_num[2], filter_num[2], kernel_size=ks, padding=padding)
        self.bn3_2 = nn.BatchNorm2d(filter_num[2])
        self.maxpool3 = nn.MaxPool2d(2)

        # Block 4
        self.conv4_1 = nn.Conv2d(filter_num[2], filter_num[3], kernel_size=ks, padding=padding)
        self.bn4_1 = nn.BatchNorm2d(filter_num[3])
        self.conv4_2 = nn.Conv2d(filter_num[3], filter_num[3], kernel_size=ks, padding=padding)
        self.bn4_2 = nn.BatchNorm2d(filter_num[3])
        self.maxpool4 = nn.MaxPool2d(2)

        # Bottleneck part of the network
        self.conv5_1 = nn.Conv2d(filter_num[3], filter_num[4], kernel_size=ks, padding=padding)
        self.bn5_1 = nn.BatchNorm2d(filter_num[4])
        self.conv5_2 = nn.Conv2d(filter_num[4], filter_num[4], kernel_size=ks, padding=padding)
        self.bn5_2 = nn.BatchNorm2d(filter_num[4])

        # Decoding part of the network

        # Block 4
        self.conv6_up = nn.ConvTranspose2d(filter_num[4], filter_num[3], kernel_size=2, stride=2)
        self.conv6_1 = nn.Conv2d(filter_num[4], filter_num[3], kernel_size=ks, padding=padding)
        self.bn6_1 = nn.BatchNorm2d(filter_num[3])
        self.conv6_2 = nn.Conv2d(filter_num[3], filter_num[3], kernel_size=ks, padding=padding)
        self.bn6_2 = nn.BatchNorm2d(filter_num[3])

        # Block 3
        self.conv7_up = nn.ConvTranspose2d(filter_num[3], filter_num[2], kernel_size=2, stride=2)
        self.conv7_1 = nn.Conv2d(filter_num[3], filter_num[2], kernel_size=ks, padding=padding)
        self.bn7_1 = nn.BatchNorm2d(filter_num[2])
        self.conv7_2 = nn.Conv2d(filter_num[2], filter_num[2], kernel_size=ks, padding=padding)
        self.bn7_2 = nn.BatchNorm2d(filter_num[2])

        # Block 2
        self.conv8_up = nn.ConvTranspose2d(filter_num[2], filter_num[1], kernel_size=2, stride=2)
        self.conv8_1 = nn.Conv2d(filter_num[2], filter_num[1], kernel_size=ks, padding=padding)
        self.bn8_1 = nn.BatchNorm2d(filter_num[1])
        self.conv8_2 = nn.Conv2d(filter_num[1], filter_num[1], kernel_size=ks, padding=padding)
        self.bn8_2 = nn.BatchNorm2d(filter_num[1])

        # Block 1
        self.conv9_up = nn.ConvTranspose2d(filter_num[1], filter_num[0], kernel_size=2, stride=2)
        self.conv9_1 = nn.Conv2d(filter_num[1], filter_num[0], kernel_size=ks, padding=padding)
        self.bn9_1 = nn.BatchNorm2d(filter_num[0])
        self.conv9_2 = nn.Conv2d(filter_num[0], filter_num[0], kernel_size=ks, padding=padding)
        self.bn9_2 = nn.BatchNorm2d(filter_num[0])

        # Output Part of Network.
        self.conv10 = nn.Conv2d(filter_num[0], output_channels, kernel_size=1)

        # Resize transformation to match the expected input size
        self.resize_transform = transforms.Resize((2 * filter_num[-1], 2 * filter_num[-1]))

    def forward(self, x):
        """ 
        Forward propagation of the network.
        """
        # Resize each image in the batch to input size
        x = self.resize_transform(x)

        # Encoding part of the network

        # Block 1
        x1 = F.relu(self.bn1_1(self.conv1_1(x)))
        x1 = F.relu(self.bn1_2(self.conv1_2(x1)))
        x1_pool = self.maxpool1(x1)

        # Block 2
        x2 = F.relu(self.bn2_1(self.conv2_1(x1_pool)))
        x2 = F.relu(self.bn2_2(self.conv2_2(x2)))
        x2_pool = self.maxpool1(x2)

        # Block 3
        x3 = F.relu(self.bn3_1(self.conv3_1(x2_pool)))
        x3 = F.relu(self.bn3_2(self.conv3_2(x3)))
        x3_pool = self.maxpool1(x3)

        # Block 4
        x4 = F.relu(self.bn4_1(self.conv4_1(x3_pool)))
        x4 = F.relu(self.bn4_2(self.conv4_2(x4)))
        x4_pool = self.maxpool1(x4)

        # Bottleneck part of the network.
        x5 = F.relu(self.bn5_1(self.conv5_1(x4_pool)))
        x5 = F.relu(self.bn5_2(self.conv5_2(x5)))

        # Decoding part of the network.

        # Block 4
        x6_up = self.conv6_up(x5)
        x6_cat = torch.cat((x4, x6_up), dim=1)  # Concatenate features from the previous layer
        x6 = F.relu(self.bn6_1(self.conv6_1(x6_cat)))
        x6 = F.relu(self.bn6_2(self.conv6_2(x6)))

        # Block 3
        x7_up = self.conv7_up(x6)
        x7_cat = torch.cat((x3, x7_up), dim=1)  # Concatenate features from the previous layer
        x7 = F.relu(self.bn7_1(self.conv7_1(x7_cat)))
        x7 = F.relu(self.bn7_2(self.conv7_2(x7)))

        # Block 2
        x8_up = self.conv8_up(x7)
        x8_cat = torch.cat((x2, x8_up), dim=1)  # Concatenate features from the previous layer
        x8 = F.relu(self.bn8_1(self.conv8_1(x8_cat)))
        x8 = F.relu(self.bn8_2(self.conv8_2(x8)))

        # Block 1
        x9_up = self.conv9_up(x8)
        x9_cat = torch.cat((x1, x9_up), dim=1)  # Concatenate features from the previous layer
        x9 = F.relu(self.bn9_1(self.conv9_1(x9_cat)))
        x9 = F.relu(self.bn9_2(self.conv9_2(x9)))

        # Output part of the network
        output = torch.sigmoid(self.conv10(x9))

        return output

class UNetPlusPlus(nn.Module):
    """ 
    UNet++ inspired architecture for semantic segmentation.
    """

    def __init__(self, filter_num, input_channels=3, output_channels=1, padding=1, ks=3):
        """ 
        Constructor for the UNetPlusPlus class.

        Parameters:
            filter_num (list): A list of the number of filters (number of input or output channels of each layer).
            input_channels (int): Input channels for the network.
            output_channels (int): Output channels for the final network.
            padding (int): Padding for convolutional layers.
            ks (int): Kernel size for convolutional layers.
        """
        # Call the constructor of the parent class
        super(UNetPlusPlus, self).__init__()

        # Encoding part of the network
        # Block 1
        self.conv1_1 = nn.Conv2d(input_channels, filter_num[0], kernel_size=ks, padding=padding)
        self.bn1_1 = nn.BatchNorm2d(filter_num[0])
        self.conv1_2 = nn.Conv2d(filter_num[0], filter_num[0], kernel_size=ks, padding=padding)
        self.bn1_2 = nn.BatchNorm2d(filter_num[0])
        self.maxpool1 = nn.MaxPool2d(2)

        # Block 2
        self.conv2_1 = nn.Conv2d(filter_num[0], filter_num[1], kernel_size=ks, padding=padding)
        self.bn2_1 = nn.BatchNorm2d(filter_num[1])
        self.conv2_2 = nn.Conv2d(filter_num[1], filter_num[1], kernel_size=ks, padding=padding)
        self.bn2_2 = nn.BatchNorm2d(filter_num[1])
        self.maxpool2 = nn.MaxPool2d(2)

        # Block 3
        self.conv3_1 = nn.Conv2d(filter_num[1], filter_num[2], kernel_size=ks, padding=padding)
        self.bn3_1 = nn.BatchNorm2d(filter_num[2])
        self.conv3_2 = nn.Conv2d(filter_num[2], filter_num[2], kernel_size=ks, padding=padding)
        self.bn3_2 = nn.BatchNorm2d(filter_num[2])
        self.maxpool3 = nn.MaxPool2d(2)

        # Block 4
        self.conv4_1 = nn.Conv2d(filter_num[2], filter_num[3], kernel_size=ks, padding=padding)
        self.bn4_1 = nn.BatchNorm2d(filter_num[3])
        self.conv4_2 = nn.Conv2d(filter_num[3], filter_num[3], kernel_size=ks, padding=padding)
        self.bn4_2 = nn.BatchNorm2d(filter_num[3])
        self.maxpool4 = nn.MaxPool2d(2)

        # Bottleneck part of the network
        self.conv5_1 = nn.Conv2d(filter_num[3], filter_num[4], kernel_size=ks, padding=padding)
        self.bn5_1 = nn.BatchNorm2d(filter_num[4])
        self.conv5_2 = nn.Conv2d(filter_num[4], filter_num[4], kernel_size=ks, padding=padding)
        self.bn5_2 = nn.BatchNorm2d(filter_num[4])

        # Decoding part of the network

        # Block 4
        self.conv6_up = nn.ConvTranspose2d(filter_num[4], filter_num[3], kernel_size=2, stride=2)
        self.conv6_1 = nn.Conv2d(filter_num[4], filter_num[3], kernel_size=ks, padding=padding)
        self.bn6_1 = nn.BatchNorm2d(filter_num[3])
        self.conv6_2 = nn.Conv2d(filter_num[3], filter_num[3], kernel_size=ks, padding=padding)
        self.bn6_2 = nn.BatchNorm2d(filter_num[3])

        # Block 3
        self.conv7_conv4_up = nn.ConvTranspose2d(filter_num[3], filter_num[2], kernel_size=2, stride=2)
        self.conv7_up = nn.ConvTranspose2d(filter_num[3], filter_num[2], kernel_size=2, stride=2)
        self.conv7_1 = nn.Conv2d(filter_num[3] + filter_num[2], filter_num[2], kernel_size=ks, padding=padding)
        self.bn7_1 = nn.BatchNorm2d(filter_num[2])
        self.conv7_2 = nn.Conv2d(filter_num[2], filter_num[2], kernel_size=ks, padding=padding)
        self.bn7_2 = nn.BatchNorm2d(filter_num[2])

        # Block 2
        self.conv8_conv3_up_1 = nn.ConvTranspose2d(filter_num[2], filter_num[1], kernel_size=2, stride=2)
        self.conv8_conv3_up_2 = nn.ConvTranspose2d(filter_num[2] + 2 * filter_num[1], filter_num[1], kernel_size=2, stride=2)
        self.conv8_up = nn.ConvTranspose2d(filter_num[2], filter_num[1], kernel_size=2, stride=2)
        self.conv8_1 = nn.Conv2d(filter_num[2] + 2 * filter_num[1], filter_num[1], kernel_size=ks, padding=padding)
        self.bn8_1 = nn.BatchNorm2d(filter_num[1])
        self.conv8_2 = nn.Conv2d(filter_num[1], filter_num[1], kernel_size=ks, padding=padding)
        self.bn8_2 = nn.BatchNorm2d(filter_num[1])

        # Block 1
        self.conv9_conv2_up_1 = nn.ConvTranspose2d(filter_num[1], filter_num[0], kernel_size=2, stride=2)
        self.conv9_conv2_up_2 = nn.ConvTranspose2d(filter_num[1] + 2 * filter_num[0], filter_num[0], kernel_size=2, stride=2)
        self.conv9_conv2_up_3 = nn.ConvTranspose2d(filter_num[1] + 4 * filter_num[0], filter_num[0], kernel_size=2, stride=2)
        self.conv9_up = nn.ConvTranspose2d(filter_num[1], filter_num[0], kernel_size=2, stride=2)
        self.conv9_1 = nn.Conv2d(filter_num[1] + 3 * filter_num[0], filter_num[0], kernel_size=ks, padding=padding)
        self.bn9_1 = nn.BatchNorm2d(filter_num[0])
        self.conv9_2 = nn.Conv2d(filter_num[0], filter_num[0], kernel_size=ks, padding=padding)
        self.bn9_2 = nn.BatchNorm2d(filter_num[0])

        # Output Part of the Network.
        self.conv10 = nn.Conv2d(filter_num[0], output_channels, kernel_size=1)

        # Resize transformation to match the expected input size
        self.resize_transform = transforms.Resize((2 * filter_num[-1], 2 * filter_num[-1]))

    def forward(self, x):
        """ 
        Forward propagation of the network.
        """
        # Resize each image in the batch to input size
        x = self.resize_transform(x)

        # Encoding part of the network

        # Block 1
        x1 = F.relu(self.bn1_1(self.conv1_1(x)))
        x1 = F.relu(self.bn1_2(self.conv1_2(x1)))
        x1_pool = self.maxpool1(x1)

        # Block 2
        x2 = F.relu(self.bn2_1(self.conv2_1(x1_pool)))
        x2 = F.relu(self.bn2_2(self.conv2_2(x2)))
        x2_pool = self.maxpool1(x2)

        # Block 3
        x3 = F.relu(self.bn3_1(self.conv3_1(x2_pool)))
        x3 = F.relu(self.bn3_2(self.conv3_2(x3)))
        x3_pool = self.maxpool1(x3)

        # Block 4
        x4 = F.relu(self.bn4_1(self.conv4_1(x3_pool)))
        x4 = F.relu(self.bn4_2(self.conv4_2(x4)))
        x4_pool = self.maxpool1(x4)

        # Bottleneck part of the network.
        x5 = F.relu(self.bn5_1(self.conv5_1(x4_pool)))
        x5 = F.relu(self.bn5_2(self.conv5_2(x5)))

        # Decoding part of the network.

        # Block 4
        x6_up = self.conv6_up(x5)
        x6_cat = torch.cat((x4, x6_up), dim=1)  # Concatenate features from the previous layer
        x6 = F.relu(self.bn6_1(self.conv6_1(x6_cat)))
        x6 = F.relu(self.bn6_2(self.conv6_2(x6)))

        # Block 3
        x7_x4_up = self.conv7_conv4_up(x4)
        x7_up = self.conv7_up(x6)
        x7_cat_1 = torch.cat((x3, x7_x4_up), dim=1)
        x7_cat_2 = torch.cat((x7_cat_1, x7_up), dim=1)
        x7 = F.relu(self.bn7_1(self.conv7_1(x7_cat_2)))
        x7 = F.relu(self.bn7_2(self.conv7_2(x7)))

        # Block 2
        x8_x3_up_1 = self.conv8_conv3_up_1(x3)
        x8_x3_up_2 = self.conv8_conv3_up_2(x7_cat_1)
        x8_up = self.conv8_up(x7)
        x8_cat_1 = torch.cat((x2, x8_x3_up_1), dim=1)
        x8_cat_2 = torch.cat((x8_cat_1, x8_x3_up_2), dim=1)
        x8_cat_3 = torch.cat((x8_cat_2, x8_up), dim=1)
        x8 = F.relu(self.bn8_1(self.conv8_1(x8_cat_3)))
        x8 = F.relu(self.bn8_2(self.conv8_2(x8)))

        # Block 1
        x9_x2_up_1 = self.conv9_conv2_up_1(x2)
        x9_x2_up_2 = self.conv9_conv2_up_2(x8_cat_1)
        x9_x2_up_3 = self.conv9_conv2_up_3(x8_cat_2)
        x9_up = self.conv9_up(x8)
        x9_cat_1 = torch.cat((x1, x9_x2_up_1), dim=1)
        x9_cat_2 = torch.cat((x9_cat_1, x9_x2_up_2), dim=1)
        x9_cat_3 = torch.cat((x9_cat_2, x9_x2_up_3), dim=1)
        x9_cat_4 = torch.cat((x9_cat_3, x9_up), dim=1)
        x9 = F.relu(self.bn9_1(self.conv9_1(x9_cat_4)))
        x9 = F.relu(self.bn9_2(self.conv9_2(x9)))

        # Output part of the network
        output = torch.sigmoid(self.conv10(x9))

        return output
'''
---------------------------------------------------------------------------------------------
Classification
---------------------------------------------------------------------------------------------
'''
class ResNet18(nn.Module):
    def __init__(self, num_classes=1, num_metadata_features=3, p_dropout=0., metadata=False):
        """
        ResNet18 model for binary classification.

        Args:
            num_classes (int): Number of output classes. Default is 1 for binary classification.
            num_metadata_features (int): Number of features in the metadata. Default is 3.
            p_dropout (float): Dropout probability. Default is 0.
            metadata (bool): Flag indicating whether metadata should be used. Default is False.
        """
        super(ResNet18, self).__init__()
        
        self.metadata = metadata

        # Load pre-trained ResNet34 model
        resnet18 = models.resnet18(weights='ResNet18_Weights.DEFAULT')

        # Feature extraction layers (remove the fully connected layers at the end)
        self.features = nn.Sequential(*list(resnet18.children())[:-2])

        # Custom fully connected layers for binary classification
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.BatchNorm1d(256),
            nn.Linear(256, num_classes),
            nn.Sigmoid()
        )

        # Metadata layers
        if self.metadata == True:
            self.metadata_fc = nn.Sequential(
                nn.Linear(num_metadata_features, 256),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Linear(256, num_classes),
                nn.Sigmoid()
            )

            self.fc_final = nn.Sequential(
                nn.Linear(2, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Linear(128, num_classes),
                nn.Sigmoid()
            )

        # Resize transformation to match the expected input size
        self.resize_transform = transforms.Resize((224, 224))

    def forward(self, x, metadata=None):
        """
        Forward pass through the ResNet18 model.

        Args:
            x (torch.Tensor): Input tensor representing the batch of images.
            metadata (torch.Tensor or None): Metadata tensor. Pass None if metadata is not used.

        Returns:
            torch.Tensor: Output tensor representing the predicted probabilities for binary classification.
        """
        # Resize each image in the batch to 224x224
        x = self.resize_transform(x)

        # Feature extraction using ResNet50
        x = self.features(x)

        # Global average pooling
        x = x.mean([2, 3])

        # Fully connected layers for classification
        x = self.fc(x)

        # If metadata is provided, concatenate it with the features
        if self.metadata == True:
            metadata_features = self.metadata_fc(metadata)
            x = torch.cat([x, metadata_features], dim=1)
            x = self.fc_final(x)
        
        # Squeeze the last dimension
        x = torch.squeeze(x, dim=1)

        return x
    
class ResNet34(nn.Module):
    def __init__(self, num_classes=1, num_metadata_features=3, p_dropout=0., metadata=False):
        """
        ResNet34 model for binary classification.

        Args:
            num_classes (int): Number of output classes. Default is 1 for binary classification.
            num_metadata_features (int): Number of features in the metadata. Default is 3.
            p_dropout (float): Dropout probability. Default is 0.
            metadata (bool): Flag indicating whether metadata should be used. Default is False.
        """
        super(ResNet34, self).__init__()
        
        self.metadata = metadata

        # Load pre-trained ResNet34 model
        resnet34 = models.resnet34(weights='ResNet34_Weights.DEFAULT')

        # Feature extraction layers (remove the fully connected layers at the end)
        self.features = nn.Sequential(*list(resnet34.children())[:-2])

        # Custom fully connected layers for binary classification
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.BatchNorm1d(256),
            nn.Linear(256, num_classes),
            nn.Sigmoid()
        )

        # Metadata layers
        if self.metadata == True:
            self.metadata_fc = nn.Sequential(
                nn.Linear(num_metadata_features, 256),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Linear(256, num_classes),
                nn.Sigmoid()
            )

            self.fc_final = nn.Sequential(
                nn.Linear(2, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Linear(128, num_classes),
                nn.Sigmoid()
            )

        # Resize transformation to match the expected input size
        self.resize_transform = transforms.Resize((224, 224))

    def forward(self, x, metadata=None):
        """
        Forward pass through the ResNet34 model.

        Args:
            x (torch.Tensor): Input tensor representing the batch of images.
            metadata (torch.Tensor or None): Metadata tensor. Pass None if metadata is not used.

        Returns:
            torch.Tensor: Output tensor representing the predicted probabilities for binary classification.
        """
        # Resize each image in the batch to 224x224
        x = self.resize_transform(x)

        # Feature extraction using ResNet50
        x = self.features(x)

        # Global average pooling
        x = x.mean([2, 3])

        # Fully connected layers for classification
        x = self.fc(x)

        # If metadata is provided, concatenate it with the features
        if self.metadata == True:
            metadata_features = self.metadata_fc(metadata)
            x = torch.cat([x, metadata_features], dim=1)
            x = self.fc_final(x)
        
        # Squeeze the last dimension
        x = torch.squeeze(x, dim=1)

        return x

class ResNet50(nn.Module):
    def __init__(self, num_classes=1, num_metadata_features=3, p_dropout=0., metadata=False):
        """
        ResNet50 model for binary classification.

        Args:
            num_classes (int): Number of output classes. Default is 1 for binary classification.
            num_metadata_features (int): Number of features in the metadata. Default is 3.
            p_dropout (float): Dropout probability. Default is 0.
            metadata (bool): Flag indicating whether metadata should be used. Default is False.
        """
        super(ResNet50, self).__init__()
        
        self.metadata = metadata

        # Load pre-trained ResNet50 model
        resnet50 = models.resnet50(weights='ResNet50_Weights.DEFAULT')

        # Feature extraction layers (remove the fully connected layers at the end)
        self.features = nn.Sequential(*list(resnet50.children())[:-2])

        # Custom fully connected layers for binary classification
        self.fc = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.BatchNorm1d(256),
            nn.Linear(256, num_classes),
            nn.Sigmoid()
        )

        # Metadata layers
        if self.metadata == True:
            self.metadata_fc = nn.Sequential(
                nn.Linear(num_metadata_features, 256),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Linear(256, num_classes),
                nn.Sigmoid()
            )

            self.fc_final = nn.Sequential(
                nn.Linear(2, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Linear(128, num_classes),
                nn.Sigmoid()
            )

        # Resize transformation to match the expected input size
        self.resize_transform = transforms.Resize((224, 224))

    def forward(self, x, metadata=None):
        """
        Forward pass through the ResNet50 model.

        Args:
            x (torch.Tensor): Input tensor representing the batch of images.
            metadata (torch.Tensor or None): Metadata tensor. Pass None if metadata is not used.

        Returns:
            torch.Tensor: Output tensor representing the predicted probabilities for binary classification.
        """
        # Resize each image in the batch to 224x224
        x = self.resize_transform(x)

        # Feature extraction using ResNet50
        x = self.features(x)

        # Global average pooling
        x = x.mean([2, 3])

        # Fully connected layers for classification
        x = self.fc(x)

        # If metadata is provided, concatenate it with the features
        if self.metadata == True:
            metadata_features = self.metadata_fc(metadata)
            x = torch.cat([x, metadata_features], dim=1)
            x = self.fc_final(x)
        
        # Squeeze the last dimension
        x = torch.squeeze(x, dim=1)

        return x

class ResNet101(nn.Module):
    def __init__(self, num_classes=1, num_metadata_features=3, p_dropout=0., metadata=False):
        """
        ResNet101 model for binary classification.

        Args:
            num_classes (int): Number of output classes. Default is 1 for binary classification.
            num_metadata_features (int): Number of features in the metadata. Default is 3.
            p_dropout (float): Dropout probability. Default is 0.
            metadata (bool): Flag indicating whether metadata should be used. Default is False.
        """
        super(ResNet101, self).__init__()
        
        self.metadata = metadata

        # Load pre-trained ResNet101 model
        resnet101 = models.resnet101(weights='ResNet101_Weights.DEFAULT')

        # Feature extraction layers (remove the fully connected layers at the end)
        self.features = nn.Sequential(*list(resnet101.children())[:-2])

        # Custom fully connected layers for binary classification
        self.fc = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.BatchNorm1d(256),
            nn.Linear(256, num_classes),
            nn.Sigmoid()
        )

        # Metadata layers
        if self.metadata == True:
            self.metadata_fc = nn.Sequential(
                nn.Linear(num_metadata_features, 256),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Linear(256, num_classes),
                nn.Sigmoid()
            )

            self.fc_final = nn.Sequential(
                nn.Linear(2, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Linear(128, num_classes),
                nn.Sigmoid()
            )

        # Resize transformation to match the expected input size
        self.resize_transform = transforms.Resize((224, 224))

    def forward(self, x, metadata=None):
        """
        Forward pass through the ResNet101 model.

        Args:
            x (torch.Tensor): Input tensor representing the batch of images.
            metadata (torch.Tensor or None): Metadata tensor. Pass None if metadata is not used.

        Returns:
            torch.Tensor: Output tensor representing the predicted probabilities for binary classification.
        """
        # Resize each image in the batch to 224x224
        x = self.resize_transform(x)

        # Feature extraction using ResNet50
        x = self.features(x)

        # Global average pooling
        x = x.mean([2, 3])

        # Fully connected layers for classification
        x = self.fc(x)

        # If metadata is provided, concatenate it with the features
        if self.metadata == True:
            metadata_features = self.metadata_fc(metadata)
            x = torch.cat([x, metadata_features], dim=1)
            x = self.fc_final(x)
        
        # Squeeze the last dimension
        x = torch.squeeze(x, dim=1)

        return x