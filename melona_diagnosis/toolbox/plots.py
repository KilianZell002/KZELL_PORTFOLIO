import torch
import torch.nn as nn
import torchvision.transforms.functional as F

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms

'''
---------------------------------------------------------------------------------------------
Metrics
---------------------------------------------------------------------------------------------
'''
def plot_loss(losses_train, losses_test, save_path=None):
    """
    Plots the training and testing losses over epochs.

    Args:
        losses_train (list): List of training losses for each epoch.
        losses_test (list): List of testing losses for each epoch.
        save_path (str): Optional. Path to save the plot. If None, the plot is displayed.
    """
    _, axs = plt.subplots(1, 1, figsize=(7.5, 5))
    epochs = range(1, len(losses_train) + 1)

    # Plot loss in the subplot
    axs.plot(epochs, losses_train, label='Train', linestyle='-')
    axs.plot(epochs, losses_test, label='Test', linestyle='-')
    axs.set_title('Loss Over Epochs')
    axs.set_xlabel('Epoch')
    axs.set_ylabel('Loss')
    axs.legend()
    axs.grid(True)

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

'''
---------------------------------------------------------------------------------------------
Displays
---------------------------------------------------------------------------------------------
'''
def plot_data_PART1(dataset, i, save_path=None):
    """
    Plots an image and its corresponding mask from the dataset.

    Args:
        dataset: Dataset object containing the samples.
        i (int): Index of the sample to be plotted.
        save_path (str): Optional. Path to save the plot. If None, the plot is displayed.
    """
    sample = dataset[i]
    image = sample['image']
    mask = sample['mask']

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(F.to_pil_image(image))
    plt.title(f"Image id: {i}")

    plt.subplot(1, 2, 2)
    plt.imshow(F.to_pil_image(mask), cmap='gray')
    plt.title(f"Mask id: {i}")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=90, bbox_inches='tight')
    else:
        plt.show()

def plot_data_PART2(traindataset, testdataset, index_train, index_test, save_path=None):
    """
    Plots images from both the training and test datasets.

    Args:
        traindataset: Training dataset object.
        testdataset: Test dataset object.
        index_train (int): Index of the sample from the training dataset to be plotted.
        index_test (int): Index of the sample from the test dataset to be plotted.
        save_path (str): Optional. Path to save the plot. If None, the plot is displayed.
    """
    sample_train = traindataset[index_train]
    sample_test = testdataset[index_test]

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(F.to_pil_image(sample_train['image']))
    plt.title(f"Input id: {sample_train['index']}, Train")

    plt.subplot(1, 2, 2)
    plt.imshow(F.to_pil_image(sample_test['image']))
    plt.title(f"Input id: {sample_test['index']}, Test")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=90, bbox_inches='tight')
    else:
        plt.show()

def show_prediction(image, mask, output, title, transparency=0.38, save_path=None):
    """
    Plots a 2x3 grid with comparisons of the output and original image.

    Args:
        image: Original image tensor.
        mask: Original mask tensor.
        output: Model output mask tensor.
        title (str): Title of the plot.
        transparency (float): Transparency value for overlaying masks.
        save_path (str): Optional. Path to save the plot. If None, the plot is displayed.
    """
    # Create a 2x3 subplot grid
    fig, axs = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(20, 15), gridspec_kw={'wspace': 0.025, 'hspace': 0.010})
    fig.suptitle(title, x=0.5, y=0.92, fontsize=20)

    # Original Mask
    axs[0][0].set_title("Original Mask", fontdict={'fontsize': 16})
    axs[0][0].imshow(F.to_pil_image(mask), cmap='gray')
    axs[0][0].set_axis_off()

    # Constructed Mask
    axs[0][1].set_title("Constructed Mask", fontdict={'fontsize': 16})
    axs[0][1].imshow(F.to_pil_image(output), cmap='gray')
    axs[0][1].set_axis_off()

    # Mask Difference
    mask_diff = torch.abs(mask - output)
    axs[0][2].set_title("Mask Difference", fontdict={'fontsize': 16})
    axs[0][2].imshow(F.to_pil_image(mask_diff), cmap='gray')
    axs[0][2].set_axis_off()

    # Original Segment
    seg_output = mask * transparency
    seg_image = (image + seg_output) / 2
    axs[1][0].set_title("Original Segment", fontdict={'fontsize': 16})
    axs[1][0].imshow(F.to_pil_image(seg_image), cmap='gray')
    axs[1][0].set_axis_off()

    # Constructed Segment
    seg_output = output * transparency
    seg_image = (image + seg_output) / 2
    axs[1][1].set_title("Constructed Segment", fontdict={'fontsize': 16})
    axs[1][1].imshow(F.to_pil_image(seg_image), cmap='gray')
    axs[1][1].set_axis_off()

    # Original Image
    axs[1][2].set_title("Original Image", fontdict={'fontsize': 16})
    axs[1][2].imshow(F.to_pil_image(image), cmap='gray')
    axs[1][2].set_axis_off()

    # Adjust layout
    plt.tight_layout()

    # Save or display the plot
    if save_path:
        plt.savefig(save_path, dpi=90, bbox_inches='tight')
    else:
        plt.show()

def show_post_processing(mask, output, processed_output, title=None, save_path=None):
    """
    Plots a 1x3 grid showing the original mask, model output, and processed output.

    Args:
        mask: Original mask tensor.
        output: Model output mask tensor.
        processed_output: Processed output mask tensor.
        title (str): Optional. Title of the plot.
        save_path (str): Optional. Path to save the plot. If None, the plot is displayed.
    """
    # Create a 1x3 subplot grid
    fig, axs = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(20, 7.5), gridspec_kw={'wspace': 0.025, 'hspace': 0.010})
    
    # Set a title if provided
    if title is not None:
        fig.suptitle(title, x=0.5, y=0.92, fontsize=20)
        
    # Original mask
    axs[0].set_title("Original mask", fontdict={'fontsize': 16})
    axs[0].imshow(F.to_pil_image(mask), cmap='gray')
    axs[0].set_axis_off()

    # Model output
    axs[1].set_title("Model output", fontdict={'fontsize': 16})
    axs[1].imshow(F.to_pil_image(output[0]), cmap='gray')
    axs[1].set_axis_off()

    # Processed output
    axs[2].set_title("Processed output", fontdict={'fontsize': 16})
    axs[2].imshow(F.to_pil_image(processed_output), cmap='gray')
    axs[2].set_axis_off()
    
    # Adjust layout
    plt.tight_layout()

    # Save or display the plot
    if save_path:
        plt.savefig(save_path, dpi=90, bbox_inches='tight')
    else:
        plt.show()