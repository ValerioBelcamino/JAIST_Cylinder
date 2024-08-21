import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from video_dataset import VideoDatasetNPY
from sequence_dataset import SequenceDatasetNPY
from utils import EarlyStopper, play_video
import matplotlib.pyplot as plt
from ViViT.vivit import ViViT
import seaborn as sns
import time
import math
import os



# _________________________________________________________________________________________________________________________
# _________________________________________________________________________________________________________________________
# _________________________________________________________________________________________________________________________


# General variables

path = '/home/s2412003/Shared/JAIST_Cylinder/Segmented_Dataset2'

sub_folders = ['Video1', 'Video2']

which_camera = 1

do_train = True

# Seed for reproducibility
np.random.seed(0)

# Initialized later
input_dim = 0
output_dim = 0
max_seq_length = 0

# Training and Evaluation
num_epochs = 200
learning_rate = 0.001
batch_size = 16

video_augmentation = False

pixel_dim = 224
patch_size = 56
max_time = 90

checkpoint_model_name = f'checkpoint_model_{sub_folders[which_camera]}_{learning_rate}lr_{batch_size}bs_{pixel_dim}px_{patch_size}ps.pt'
confusion_matrix_name = f'confusion_matrix_{sub_folders[which_camera]}_{learning_rate}lr_{batch_size}bs_{pixel_dim}px_{patch_size}ps.png'

print(f'Saving model to {checkpoint_model_name}')
print(f'Saving confusion matrix to {confusion_matrix_name}')

checkpoit_model_name = os.path.join(path, checkpoint_model_name)
confusion_matrix_name = os.path.join(path, confusion_matrix_name)


action_names = ['linger', 'massaging', 'patting', 
                'pinching', 'press', 'pull', 
                'push', 'rub', 'scratching', 
                'shaking', 'squeeze', 'stroke', 
                'tapping', 'trembling']

action_dict = {action: i for i, action in enumerate(action_names)}
action_dict_inv = {i: action for i, action in enumerate(action_names)}

action_cut_time_dict = {'linger': 5, 'massaging': 2, 'patting': 3,
                        'pinching': 3, 'press': 2, 'pull': 4,
                        'push': 4, 'rub': 2, 'scratching': 2,
                        'shaking': 2, 'squeeze': 4, 'stroke': 3,
                        'tapping': 2, 'trembling': 2}


# _________________________________________________________________________________________________________________________
# _________________________________________________________________________________________________________________________
# _________________________________________________________________________________________________________________________

video_filenames = []
video_labels = []
imu_filenames = []

sub_folder = None
if which_camera == 0:
    sub_folder = sub_folders[0]
elif which_camera == 1:
    sub_folder = sub_folders[1]

print(f'Using {sub_folder}...')

trials = os.listdir(path)
for trial in sorted(trials):
    video_names = sorted([os.path.join(path, trial, sub_folder, f) for f in os.listdir(os.path.join(path, trial, sub_folder))])
    video_filenames.extend(video_names)
    imu_names = sorted([os.path.join(path, trial, 'IMU', f) for f in os.listdir(os.path.join(path, trial, 'IMU'))])
    imu_filenames.extend(imu_names)

lenghts = []

for name in imu_filenames:
    name = os.path.basename(name)
    lenghts.append(name.split('_')[2])

print(f'{len(video_filenames)}')
print(f'{len(imu_filenames)}')

# Check that there are no duplicates
assert len(video_filenames) == len(set(video_filenames)) == len(imu_filenames)




X_train_names, X_test_names, Y_train_labels, Y_test_labels = train_test_split(
                                            video_filenames,
                                            range(len(video_filenames)), 
                                            test_size=0.3, 
                                            random_state=0
                                            )

X_train_imu_names, X_test_imu_names, Y_train_labels, Y_test_labels = train_test_split(
                                            imu_filenames,
                                            range(len(video_filenames)), 
                                            test_size=0.3,
                                            random_state=0
                                            )

print(f'\n{len(X_train_names)=}, {len(X_test_names)=}')
print(f'{len(Y_train_labels)=}, {len(Y_test_labels)=}\n')

train_dataset = VideoDatasetNPY(X_train_names, Y_train_labels, [], max_length=max_time, pixel_dim=pixel_dim, video_augmentation=video_augmentation)
test_dataset = VideoDatasetNPY( X_test_names,  Y_test_labels,  [], max_length=max_time, pixel_dim=pixel_dim, video_augmentation=video_augmentation)
print('Datasets Initialized\n')

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print('Data Loaders Initialized\n')

train_dataset_imu = SequenceDatasetNPY(X_train_imu_names, Y_train_labels, lenghts, max_len=90)
train_loader_imu = DataLoader(train_dataset_imu, batch_size=batch_size, shuffle=True)
print('\n\n\n')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

imu_list = []
count = 0
if do_train:
    for epoch in range(1):
        for imus, _, lenghts in train_loader_imu:


            print(f'Batch {count} over {len(train_loader)}')
            count += 1
            imus = imus.to(device)
            for i in range(imus.shape[0]):
                element = imus[i, :int(lenghts[i])]
                element = element.cpu().numpy()
                imu_list.append(element)
            
print(f'{len(imu_list)}')
# now let's stack all the elements
stacked_imus = np.concatenate(imu_list, axis=0)
print(f'{stacked_imus.shape}')

#let's compute the mean and std for each feature
mean = np.mean(stacked_imus, axis=0)
std = np.std(stacked_imus, axis=0)
print(f'{mean=}, {std=}')

# let's compute max and min for each feature
max_values = np.max(stacked_imus, axis=0)
min_values = np.min(stacked_imus, axis=0)
print(f'Maximum values per feature: {max_values}')
print(f'Minimum values per feature: {min_values}')

# print to file these values

# with open(f'{os.path.basename(path)}.txt', 'w') as f:
#     f.write('\nMean\n')
#     f.write(f'{mean}')
#     f.write('\nStd\n')
#     f.write(f'{std}')
#     f.write('\nMax\n')
#     f.write(f'{max_values}')
#     f.write('\nMin\n')
#     f.write(f'{min_values}')

video_list = []
count = 0

summ = 0
summ_squared = 0
total_pixels = 0

if do_train:
    for epoch in range(1):
        for videos, labels in train_loader:
            print(f'Batch {count} over {len(train_loader)}')
            count += 1
            videos, labels = videos.to(device), labels.to(device)

            # print(videos.shape)
            for i in range(videos.shape[0]):
                for j in range(videos.shape[1]):

                    total_pixels += videos[i, j, 0, :, :].cpu().numpy().size
                    summ += np.sum(videos[i, j, 0, :, :].cpu().numpy())


        # Calculate mean
        mean = summ / total_pixels

        for videos, labels in train_loader:
            print(f'Batch {count} over {len(train_loader)}')
            count += 1
            videos, labels = videos.to(device), labels.to(device)

            # print(videos.shape)
            for i in range(videos.shape[0]):
                for j in range(videos.shape[1]):
                    summ_squared += np.sum((videos[i, j, 0, :, :].cpu().numpy() - mean) ** 2)

        std = np.sqrt(summ_squared / total_pixels)


            # for i in range(videos.shape[0]):
            #     video_list.append(videos[i].cpu().numpy())


# # Compute the mean and std of the videos
# video_list = np.array(video_list)
# mean = np.mean(video_list)
# std = np.std(video_list)


# Calculate variance
# variance = (summ_squared / total_pixels) - (mean ** 2)

# # Calculate standard deviation
# std = np.sqrt(variance)

print(f'{mean=}, {std=}')
exit()
