# ---------------------------------------------
# THIS ALGORITHM WAS CREATED TO TRAIN A MODEL TO IDENTIFY IMAGES
# ORIGINAL DEVELOPER: SANDEEP JILLA OF TAMUCC
# LAST UPDATED ON 10/08/2024 BY: DEANA CROUSER
# CONTACT: DEANACROUSER@GMAIL.COM
# ---------------------------------------------

from __future__ import print_function, division  # Importing modules for compatibility with Python 2.x

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from PIL import Image
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torchvision.models import resnet50, ResNet50_Weights
import matplotlib.pyplot as plt
import time
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import copy
from torch.utils import data
import seaborn as sn
import pandas as pd
from tqdm import tqdm
plt.ion()   # interactive mode

##########################################
# update folder path here:
folder_path = r'/Users/enrique.montes/Desktop/uw_classifier/randomized_balanced/'
##########################################

model_name = folder_path.replace("/", "")  # Creating model name based on folder path

# Loading data (added "if __name__ == '__main__':" to prevent code from running on import)
if __name__ == '__main__':
    print('loading data...')
    dataset = datasets.ImageFolder(folder_path, transform=transforms.Compose(
                                [transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor()]))

    # Creating data loader
    loader = data.DataLoader(dataset,
                            batch_size=10,
                            num_workers=0,
                            shuffle=False)

    print("done loading data")

    # Calculating mean and standard deviation for normalization
    print('calculating mean and stardard deviation for normalization...')
    mean = 0.0
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
    mean = mean / len(loader.dataset)

    var = 0.0
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        var += ((images - mean.unsqueeze(1))**2).sum([0,2])
    std = torch.sqrt(var / (len(loader.dataset)*224*224))

    # Predefined mean and standard deviation values
    mean,std =[[0.1195, 0.1047, 0.0819],[0.23, 0.2075, 0.1644]]

    # Data transformation
    transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    # Creating image dataset with transformation
    print('creating image dataset with transformation...')
    images = datasets.ImageFolder(os.path.join(folder_path),transform)

    # Splitting dataset into train, validation, and test sets
    print('splitting dataset into train, validation, and test sets...')
    data_size = len(images.imgs)
    train_size = int(data_size*0.8)
    val_size = (data_size-train_size)//2
    test_size = data_size-train_size-val_size
    train_set, val_set, test_set = data.random_split(images, (train_size,val_size,test_size))
    image_datasets = {'train':train_set, 'val':val_set,'test':test_set}

    # Creating data loaders
    print('creating data loaders...')
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32,
                                                shuffle=True, num_workers=4)
                for x in ['train', 'val','test']}

    # Dataset sizes
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val','test']}
    class_names = images.classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset_sizes

    # Function to display images
    def imshow(inp, title=None):
        """Imshow for Tensor."""
        inp = inp.numpy().transpose((1, 2, 0))
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        if title is not None:
            plt.title(title)

    # Get a batch of training data
    inputs, classes = next(iter(dataloaders['train']))
    print("sample batch training data:", inputs[0][0].shape)

    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)
    plt.figure(figsize=(20,5))
    imshow(out, title=[class_names[x] for x in classes])
    plt.savefig(model_name+'input samples.png')

    print("done saving input samples.png")

    # Function to train the model
    def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model

    # Function to visualize the model
    def visualize_model(model, num_images=40):
        was_training = model.training
        model.eval()
        images_so_far = 0
        fig = plt.figure(figsize=(20,20),dpi=200)
        sm = torch.nn.Softmax()
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(dataloaders['test']):
                inputs = inputs.to(device)
                l=labels
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                confidence = np.amax(np.array(sm(outputs.cpu())),1)
    #             break
                for j in range(inputs.size()[0]):
                    images_so_far += 1
                    ax = plt.subplot(num_images//4, 4, images_so_far)
                    ax.axis('off')
                    ax.set_title('{0}; GT: {1}; Confidence {2:.2f}'.format(class_names[preds[j]],class_names[l[j]],confidence[j]))
    #                 imshow(inputs.cpu().data[j])
                    inp = inputs.cpu().data[j].numpy().transpose((1, 2, 0))

                    inp = std * inp + mean
                    inp = np.clip(inp, 0, 1)
                    ax.imshow(inp)
                    if images_so_far == num_images:
                        model.train(mode=was_training)
                        return
            model.train(mode=was_training)

    # Loading the pre-trained ResNet50 model
    print('loading the pre-trained model...')
    model_ft = models.resnet50(weights=ResNet50_Weights.DEFAULT) # Updated 10/8/24
    num_ftrs = model_ft.fc.in_features

    # Modifying the final layer to match the number of classes
    model_ft.fc = nn.Linear(num_ftrs, len(class_names))

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 5 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)

    # Training the model
    print('training the model...')
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                        num_epochs=15)
    print('done training')
    # Visualizing model predictions
    visualize_model(model_ft)
    plt.savefig(model_name+'.png')


    # Saving the trained model
    print('saving the model...')
    torch.save(model_ft, model_name+'_model.pth')
    print('done saving')

    # Loading the saved model
    print('loading model..')
    model_ft = torch.load(model_name+'_model.pth')

    ############################
    # ResNet Result analysis
    ############################

    print('running ResNet result analysis...')

    nb_classes = len(class_names)

    confusion_matrix = torch.zeros(nb_classes, nb_classes)
    with torch.no_grad():
        for i, (inputs, classes) in enumerate(dataloaders['test']):
            inputs = inputs.to(device)
            classes = classes.to(device)
            outputs = model_ft(inputs)
            _, preds = torch.max(outputs, 1)
            for t, p in zip(classes.view(-1), preds.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1

    print(confusion_matrix)

    df_cm = pd.DataFrame(confusion_matrix.numpy().astype('int'), index = [i for i in class_names],
                    columns = [i for i in class_names])

    test_acc=np.trace(confusion_matrix.numpy().astype('int'))/np.sum(np.sum(df_cm))
    df_cm.to_csv(model_name+'.csv')
    plt.figure(figsize = (len(class_names)//2,len(class_names)//2))
    plt.title('Confusing Matrix of Ground-truth (row) and Prediction (column) with Testing Accuracy: {:0.2f}%'.format(test_acc*100))
    sn.heatmap(df_cm, annot=True,fmt='g',cmap="YlGnBu")
    plt.savefig(model_name+'Confusing Matrix.png')

    nor = df_cm / np.array([df_cm.sum(axis=1)]*len(class_names)).transpose()
    plt.figure(figsize = (len(class_names)//2,len(class_names)//2))
    plt.title('Normalized Confusing Matrix (%) of Ground-truth (row) and Prediction (column) with Testing Accuracy: {:0.2f}%'.format(test_acc*100))
    sn.heatmap(nor*100, annot=True,fmt='.0f',cmap="Greens")
    plt.savefig(model_name+'Confusing Matrix(%).png')

    print('process complete!')
