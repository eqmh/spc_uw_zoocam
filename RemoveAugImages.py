# ---------------------------------------------
# THIS CODE REMOVES ALL IMAGES IN A GIVEN DIRECTORY OR FOLDER
# WITH "_AUGMENTED" IN THE FILE NAME
# LAST UPDATED ON 2/22/2024 BY: DEANA CROUSER
# CONTACT: DEANACROUSER@GMAIL.COM
# ---------------------------------------------

import os

###############################################
# SET THE DIRECTORY PATH HERE:
dir_name = r"D:\NOAA GPU Hackathon\ImageTrainingDataset"
###############################################

# Function to remove augmented images from a directory
def remove_augmented_images(directory):
    # Create a list to hold the file paths of images
    files_list = []

    # Add list of files to the files_list
    for file_name in os.listdir(directory):
        # Append the absolute path of each file to files_list
        file_path = os.path.join(directory, file_name)
        if os.path.isfile(file_path) and 'augmented' in file_name:
            files_list.append(file_path)

    # Remove the augmented images
    for image in files_list:
        print("Removing:", image)
        os.remove(image)

# Check if the specified directory contains subdirectories
if any(os.path.isdir(os.path.join(dir_name, name)) for name in os.listdir(dir_name)):
    # Iterate over each subdirectory in the specified directory
    for root, subdirectories, files in os.walk(dir_name):
        for subdirectory in subdirectories:
            # Get the absolute path of the current directory
            workingDir = os.path.join(root, subdirectory)
            print("Working directory:", workingDir)
            # Remove augmented images from the current directory
            remove_augmented_images(workingDir)
else:
    # If the specified directory does not contain subdirectories,
    # remove augmented images directly from it
    remove_augmented_images(dir_name)

print("Done")

