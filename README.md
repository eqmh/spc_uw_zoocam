# ImagePreprocessing
Preprocessing steps to perform on the Image Training Dataset prior to algorithm training and classification.  
Included is a variety of python scripts to aid in augmentation and balancing in order to prepare your dataset for algorithm training.

# Getting Started
For best performance, it is recommended that image training datasets consist of augmented and balanced categories in order to avoid algorithm bias. Begin with downloading and extracting the dataset (ImageTrainingDataset.zip) file using the shared drive link provided in the Image Training Dataset Repository README file.

# Step1: dataAugmentation.py 
Run this script in order to expand the provided image dataset and make it more robust.  
This script will augment each image 5 different ways (rotate, shear, zoom, flip and brighten).  
If you start with 100 images, you will end with 600 images (500 of which will have an "_augmented" extension added to the file name)  
Update "dir_path" on (line 12) with the correct directory path where you extracted the ImageTrainingDataset. Note that the path you choose for "dir_path" can either be a directory that contains multiple folders OR it can be the PATH to one specific folder you want to augment. 

# Step2: RandomImageSelection_1.2.py
Run this scrip after dataAugmentation.py in order to create a balanced training dataset.
This script will take a random subsample of images (originial and augmented) and move it to a new directory.
To determine the number of random images to select, identify your limiting category (ie. the category containing the least number of images).  

For example, in the Image Training Dataset, Anthomedusae - (Euphysa tentaculata) has 38 originial images and is the limiting category. After data augmentation, there should be about 266 images. This is the random number of images you should collect from each category in order to maintain a balanced training dataset. 


