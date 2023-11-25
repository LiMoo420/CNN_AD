'''
# This part of the code is used to successfully read the image data from the pre-processed data folder 
# and retain the label, classification (no dementia, very mild, mild, severe dementia)

# Author: Bo Yin[MC36455*] & Zihan Xue[MC36588*]
# Contact: mc36455@um.edu.mo For Mr.Bo Yin
#          mc36588@um.edu.mo For Ms.Zihan Xue
'''

#==============================================================================
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import random
import numpy as np
import re
import os
import glob
#==============================================================================

def if_plot(filedir):
    # Define the categories and the number of images per category we want to plot
    categories = ['non_demented', 'verymild_demented', 'mild_demented', 'moderate_demented']
    images_per_category = 2  # Each category will have 2 images displayed vertically

    fig, axes = plt.subplots(nrows = 2, ncols = 4, figsize = (15, 6))     # Set up the matplotlib figure and axes
    fig.suptitle('Sample images from 4 categories', fontsize = 16)
    plt.subplots_adjust(hspace = 0.4, wspace = 0.1)     # Remove spacing between images
    for col, category in enumerate(categories):         # Iterate over each category and each item within the category
        category_dir = os.path.join(filedir, 'train', category)
        images = os.listdir(category_dir)
        selected_images = random.sample(images, images_per_category)

        for row, image in enumerate(selected_images):
            image_path = os.path.join(category_dir, image)
            img = Image.open(image_path).convert('L')
            ax = axes[row, col]
            ax.imshow(img, cmap = 'gray')
            ax.axis('off')
            if row == 0:
                ax.set_title(category.replace('_', ' ').title())
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def preprocess_image(image_path, shape = (128, 128)):
    """
    # Preprocessing the MRI images of the brain.
    #
    # Input
    # =====
    # `image_path`: (str) The file path of the image to be processed.
    # `shape`: (tuple) Reshape size.
    #
    # Return
    # =====
    # `img_array`: (ndarray) The normalized value of a single image.
    """
    img = Image.open(image_path).convert('L')       # Convert to grayscale if needed
    resized_img = img.resize(shape)         # Resize the image to make convolution easier
    enhancer = ImageEnhance.Contrast(resized_img)
    enhanced_img = enhancer.enhance(2.0)        # Factor 2.0 will increase contrast
    img_array = np.array(enhanced_img) / 255.0          # Normalizes the image array

    return img_array


def preprocess_dataset(input_data):
    """
    # The given image data set is preprocessed.
    # 
    # Parameters:
    # =====
    # Input_data (list): A two-dimensional list containing image names and paths.
    #
    # Return:
    # =====
    # list: A list of preprocessed image arrays and labels.
    """
    classifications = ['non', 'verymild', 'mild', 'moderate']
    processed_data = []         # Initialize the list of processed data
    for class_data in input_data:           # Walk through each category
        for item in class_data:             # Walk through each image in each category
            image_name, image_path = item           # Get image name and path
            image_array = preprocess_image(image_path, shape = (128, 128))            # Preprocess the image
            label = image_name.split('_')[0]            # Extract classification tags for images
            one_hot_label = [1 if label == classification else 0 for classification in classifications]
            processed_data.append((image_array, one_hot_label))         # Adds image arrays and labels to the list
    return processed_data


def trans_img_to_data(*datasets):
    """
    # Multiple image data sets are preprocessed and their processing results are returned.
    #
    # Parameters:
    # =====
    # Datasets (tuple): A tuple containing multiple image datasets.
    # 
    # Return:
    # =====
    # list: A list containing all the processed data sets.
    """
    processed_datasets = []
    for dataset in datasets:        # Iterate through each data set passed in
        processed_data = preprocess_dataset(dataset)            # Preprocess the current data set
        processed_datasets.append(processed_data)       # Add the processed data set to the list
    return processed_datasets


def load_data(filedir, splits = [], classifications = [], transform = True, ifplot = True):
    """
    # Load the lung X-Ray picture data of the classification.
    #
    # Input
    # =====
    # `filedir`: (str) The directory of splited data.
    # `splits`: (list) Different uses of data.
    # `classifications`: (list) Classification of disease degree.
    #
    # Return
    # =====
    # `data_list[0]`: (list) A list of imgaes infomations in train.
    # `data_list[1]/[2]`
    """
    if ifplot:
        if_plot(filedir)
    # Creates a 3D list to store all image names and image paths
    data_list = [[[] for _ in range(len(classifications))] for _ in range(len(splits))]
    for i, split in enumerate(splits):      # Walk through each data segmentation and classification
        for j, cls in enumerate(classifications):
            sub_dir = os.path.join(filedir, split, cls + '_demented')       # Build the path to the subdirectory
            images = sorted(
                glob.glob(os.path.join(sub_dir, '*.jpg')),
                key = lambda x: int(re.search(r'\d+', x).group()))   # Get the path to all image files
            for img_path in images:         # Add image name and path to data list
                data_list[i][j].append([os.path.basename(img_path), img_path])
    if transform:
        return trans_img_to_data(data_list[0], data_list[1], data_list[2])
    else: return data_list

#-------------------------------Last Edited By Borris at 13:10 on 21/11/2023-------------------------------------