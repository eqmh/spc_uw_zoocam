#--------------------------------
# THIS SCRIPT RANDOMLY SELECTS A PRE-DETERMINED NUMBER OF IMAGES FROM A SOURCE
# AND COPIES THE IMAGE TO A NEW OUTPUT FOLDER IN ORDER TO CREATE A BALANCED TRAINING DATASET
# LAST UPDATED ON 2/22/2024 BY: DEANA CROUSER
# CONTACT: DEANACROUSER@GMAIL.COM
#---------------------------------

# IMPORT NECESSARY PACKAGES
import os
import random
import shutil

##############################################################################
# identify source and destination path for images to be copied to and from
source = r"/Users/enrique.montes/Desktop/uw_classifier/training_dataset_augmented/"
destPath = r"/Users/enrique.montes/Desktop/uw_classifier/randomized_balanced/"
##############################################################################

# RUN THROUGH EACH SUBDIRECTORY IN THE SOURCE DIRECTORY
for root, subdirectories, files in os.walk(source):
    for subdirectory in subdirectories:
        # GET TEH ABSOLUTE PATH OF THE CURRENT SUBDIRECTORY
        workingDir = os.path.join(source, subdirectory)
        # CHANGE THE CURRENT WORKING DIRECTORY TO THE SUBDIRECTORY
        os.chdir(workingDir)
        print("working directory:", os.getcwd())

        # CREATE AN EMPTY LIST TO HOLD THE FILE PATHS OF .JPG, .PNG, AND .JPEG IMAGES
        files_list = []

        # WHILE IN EACH SUBDIRECTORY, SEARCH FOR ALL FILES THAT END WITH .JPG, .PNG, AND .JPEG
        for root, subdirectories, files in os.walk(workingDir):
            for file in files:
                # CHECK IF THE FILE ENDS WITH .JPG, .PNG, OR .JPEG AND APPEND ITS PATH TO FILES_LIST
                if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg"):
                  files_list.append(os.path.join(root, file))

            # COUNT AND PRINT THE NUMBER OF .JPG, .PNG, AND .JPEG IMAGES 
            file_count = len(files_list)
            print(file_count)

            # PRINT RANDOM FILES FROM FILES_LIST TO BE COPIED
            filesToCopy = random.sample(files_list, 300) # SET NUMER OF RANDOM IMAGES HERE (200 IN ORIGINAL CODE)
            print(filesToCopy)

            # CREATE A FINAL DESTINATION PATH BASED ON THE SUBDIRETORY NAME
            for file in filesToCopy:
                fileFolder = os.path.dirname(file)
                folderDest = os.path.basename(fileFolder)
                finalPath = os.path.join(destPath, folderDest)
                # print(finalPath) # UNCOMMENT IF YOU WANT TO SEE THE FINAL PATH

            # IF THE DESTINATION DIRECTOY DOES NTO EXIST, CREATE IT
            if os.path.isdir(finalPath) == False:
                os.makedirs(finalPath)

            # ITERATE OVER ALL RANDOM FILES AND MOVE THEM TO THE FINAL DESTINATION
            for file in filesToCopy:
                shutil.copy(file, finalPath)
                
# PRINT "DONE" AFTER ALL OPERATIONS ARE COMPLETED
print("Done")