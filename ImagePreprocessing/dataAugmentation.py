# ---------------------------------------------
# THIS CODE WAS CREATED TO AUGMENT IMAGES / EXPAND IMAGE DATASETS
# LAST UPDATED ON 10/8/2024 BY: DEANA CROUSER
# CONTACT: DEANACROUSER@GMAIL.COM
# ---------------------------------------------
from PIL import Image, ImageFilter
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

###############################################
# SET THE DIRECTORY PATH HERE:
dir_path = r"PATH TO DIRECTORY OR FOLDER HERE"
###############################################

# Function to augment images in a directory
def augment_images_in_directory(directory):
    # Get the list of files in the directory
    files = os.listdir(directory)

    # Flag to check if any image files are found
    found_images = False

    # Iterate over each file in the directory
    for file_name in files:
        file_path = os.path.join(directory, file_name)

        # Check if the item is a file and is an image file (assumed to have extensions jpg, jpeg, or png)
        if os.path.isfile(file_path) and file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            found_images = True
            # Load the image
            img = load_img(file_path)
            x = img_to_array(img)
            x = x.reshape((1,) + x.shape)

            # Define an image data generator with augmentation parameters
            datagen = ImageDataGenerator(
                rotation_range=40,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                vertical_flip=True,
                brightness_range=(0.5, 1.5))

            # Generate and save augmented samples
            prefix = os.path.splitext(file_name)[0] + '_augmented'
            i = 0
            for batch in datagen.flow(x, batch_size=1, save_to_dir=directory, save_prefix=prefix, save_format='png'):
                i += 1
                if i >= 5:  # Generate 5 augmented images per original image
                    break

    # If no image files were found in the directory, print a message
    if not found_images:
        print("No image files found in directory:", directory)


# Check if the specified directory contains subdirectories
if os.path.isdir(dir_path):
    # Iterate over each directory in the main directory
    for root, dirs, files in os.walk(dir_path):
        for directory in dirs:
            print("Processing directory:", directory)
            directory_path = os.path.join(root, directory)
            augment_images_in_directory(directory_path)
    # Augment images in the provided directory if it contains image files
    augment_images_in_directory(dir_path)
else:
    # If there are no subdirectories, perform actions on the provided directory itself
    print("No subdirectories found. Processing the provided directory:", dir_path)
    augment_images_in_directory(dir_path)

print("Done")
