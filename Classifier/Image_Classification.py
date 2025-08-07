# Import Libraries
import torch
#from torch.utils import data
#import torchvision
from torchvision import datasets, models, transforms
#from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
import cv2
import pandas as pd
from PIL import Image
import multiprocessing

if __name__ == '__main__':
    multiprocessing.freeze_support()

    #####################################
    # Input location & output directory
    unclassified_location = r"/Users/enrique.montes/Desktop/uw_classifier/unclassified_dataset/"
    out_dir = r"/Users/enrique.montes/Desktop/uw_classifier/classified_output/"
    model_location = r"/Users/enrique.montes/Desktop/uw_classifier/models/2025_08_05/Usersenrique.montesDesktopuw_classifierrandomized_balanced_model.pth"
    #####################################

    # Check if CUDA is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Set batch size
    batch_size = 32

    print("device:", device)  # Output: cuda:0 if GPU is available, else cpu

    # Count the number of PNG files in the unclassified location
    print('counting the number of files in the unclassified location...')
    count = 0
    for root, dirs, files in os.walk(unclassified_location):
        for file in files:
            if file.endswith(".png"):
                count += 1
    print("count:", count)  # Output: Total count of PNG files

    mean, std = [[0.1195, 0.1047, 0.0819], [0.2350, 0.2075, 0.1644]]

    # Data transformations for input images
    print('transforming images...')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Load images from unclassified_location and create DataLoader for inference
    print("loading images from unclassified location and creating dataloader for inference...")
    images = datasets.ImageFolder(unclassified_location, transform=transform)
    inf_dl = torch.utils.data.DataLoader(images, batch_size=32, shuffle=False, num_workers=4)
    print('done loading images')  # Output: done

    # Extract file paths for images
    print("extracting file paths for images...")
    file_paths = []
    for i in range(len(images)):
        file = images.imgs[i][0].replace("\\", "/")
        file_paths.append(file)

    print("file path:", file_paths[:5])  # Output: List of file paths
    print("total unique file paths:", len(np.unique(file_paths)))  # Output: Total unique file paths

    # Inference Function
    print('defining inference function...')


    def infer(model, dataloader):
        sm = torch.nn.Softmax()
        all_preds = []
        with torch.no_grad():
            for i, (inputs, classes) in enumerate(dataloader):
                inputs = inputs.to(device)
                classes = classes.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.append(preds)

        return all_preds


    print('done')  # Output: done

    # Define class names for prediction outputs
    print('defining class names for prediction outputs...')
    # class_names = ['Anthomedusae - (Euphysa tentaculata)', 'Calanoida - (Acartia spp.)', 'Calanoida - (Calanus sp.)',
    #                'Calanoida - (Centropages abdominalis)', 'Calanoida - (Metridia spp.)',
    #                'Calanoida - (Psudo Micro Para)',
    #                'Chaetognatha', 'ClusteredSnow', 'Copepoda - (nauplii)', 'Cyclopoida - (Oithona spp.)',
    #                'Cydippida - (Euplokamis dunlapae)',
    #                'Cydippida - (Pleurobrachia bachei)', 'Cydippida - (Unknown)',
    #                'Cyphocaridae - (Cyphocaris challengeri)',
    #                'Decapoda - Caridea (Shrimp)', 'Diatoms', 'Dinoflagellata - (Noctiluca)', 'Eggs',
    #                'Euphausiacea - Euphausiidae (Krill)',
    #                'Filament_Filaments', 'Fish_larvae', 'Gammeridea- (possibly Calliopius sp)',
    #                'Harpacticoida - (Microsetella rosea)',
    #                'Hyperiidea - (Themisto pacifica _ Hyperoche sp.)', 'Larvacea - (Oikopleura dioica)', 'Lobata',
    #                'MarineSnow',
    #                'Ostracoda - (Halocyprididae)', 'Poecilostomatoida - (Ditrichocoryceaus anglicus)',
    #                'Poecilostomatoida - (Triconia spp.)',
    #                'Pteropoda - (Clione limacina)', 'Pteropoda - (Limacina helicina)',
    #                'Siphonophore - Calycophorae (Muggiaea atlantica)',
    #                'Trachymedusae - (Aglantha digitale)', 'Trachymedusae - (Pantachogon haeckeli)',
    #                'Trachymedusae - (young)', 'Unknown']

    class_names = ['Acantharea', 'Centric', 'Ceratium', 'Chaetoceros', 'Chaetognaths', 'Chain2', 'Chain3',
                   'Copepods', 'Decapods', 'Detritus','Echinoderms', 'Guinardia', 'Jellies', 'Jellies_2','Larvaceans',
                   'Nauplia','Neocalyptrella', 'Noctiluca', 'Ostracods', 'Polychaets', 'Pteropods', 'Tricho', 'bubbles', 'pellets']

    # Run Inference
    print("running inference...")
    model = torch.load(model_location)
    model.eval()
    preds = infer(model, inf_dl)
    print('done')  # Output: done

    # Count total number of predictions
    pred_count = 0
    for batch in preds:
        for i in batch:
            pred_count += 1
    print("total number of predictions: ", pred_count)  # Output: Total number of predictions

    # Verify data availability
    print("Number of predictions:", len(preds))
    print("Number of file paths:", len(file_paths))

    # Check if both preds and file_paths contain data
    if len(preds) == 0 or len(file_paths) == 0:
        print("No data available in preds or file_paths. Exiting...")
        exit()  # Exit the script if no data is available

    # Creating the CSV output
    from datetime import datetime
    from os import path, mkdir
    from shutil import copyfile
    import re

    print('creating the csv output....')


    def get_contours(fname):
        img = np.array(Image.open(fname).convert('L'))
        shape = img.shape
        ret, thresh = cv2.threshold(img.astype(np.uint8), 100, 255, cv2.THRESH_TRUNC)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        # find the biggest countour (c) by the area
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        return x, y, w, h, shape


    def move_files(fileloc, pred_class):
        class_folder = "{}/{}".format(out_dir, pred_class)
        out_location = "{}/{}".format(class_folder, fileloc.split("/")[-1])
        if not path.isdir(class_folder):
            mkdir(class_folder)  # check if class folder exists, if not create it.
        if path.isfile(out_location):
            print("Skipping! this file already exists in the output folder: {}".format(fileloc.split("/")[-2:]))
        else:
            copyfile(fileloc, out_location)


    def get_unix(name):
        unix_lst = [word for word in re.split("-|_", name) if len(word) == 16]
        for item in unix_lst:
            if item.isdecimal():
                return (int(item) / 1000000)


    def make_csv(preds, file_paths):
        if not path.isdir(out_dir):
            mkdir(out_dir)  # check if output folder exists, if not create it.

        df_data = []
        batch_number = 0
        for p_batch in preds:
            for i, inf in enumerate(p_batch):
                try:
                    image_number = i + batch_number * batch_size  # finds corresponding name for the file
                    file_loc = file_paths[image_number]
                    filename = file_paths[image_number].split("/")[-1]
                    # unix = get_unix(filename) # # Pieter's fix
                    # date_time = datetime.utcfromtimestamp(unix).strftime('%Y-%m-%d %H:%M:%S') # Pieter's fix
                    date_time = pd.to_datetime(filename[:19], format='%Y%m%d_%H%M%S.%f')  # Pieter's addition
                    x, y, w, h, shape = get_contours(file_loc)
                    df_data.append({
                        "file_name": filename,
                        # "date": date_time.split(" ")[0], # Pieter's fix
                        # "time": date_time.split(" ")[1], # Pieter's fix
                        "date": date_time.date(),  # Pieter's addition
                        "time": date_time.time(),  # Pieter's addition
                        "predicted_class": class_names[int(inf)],
                        "x": x,
                        "y": y,
                        "w": w,
                        "h": h,
                        "image_size": shape
                    })
                    move_files(file_loc, class_names[int(inf)])  # moves file to its predicted class
                except Exception as e:
                    print(image_number)
                    print(file_loc)
                    print(e)
            batch_number += 1  # increase the batch counter by 1

        df_outputs = pd.DataFrame(df_data)
        df_outputs.to_csv(path.join(out_dir, "predictions.csv"), index=False)  # Set index=False to exclude row indices
        print("output csv shape: ", df_outputs.shape)
        print(df_outputs.head())

    import time

    start = time.time()
    make_csv(preds, file_paths)
    end = time.time()
    print("created the output folders in: ", end - start, "seconds")

    # Visualise the contours
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches


    def get_contours(fname, show_img=False):
        img = np.array(Image.open(fname).convert('L'))
        fig, ax = plt.subplots()
        ax.imshow(img, cmap='gray')
        plt.show()

        ret, thresh = cv2.threshold(img.astype(np.uint8), 100, 255, cv2.THRESH_TRUNC)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        # find the biggest countour (c) by the area
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)

        if show_img:
            fig, ax = plt.subplots()
            ax.imshow(img, cmap='gray')
            rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            plt.show()
        print(img.size)


    image_num = 10
    get_contours(file_paths[image_num], True)

    print('Classification Complete!!')




