import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from models import HAR_Transformer
from sequence_dataset import SequenceDataset
from video_dataset import VideoDataset
from utils import EarlyStopper, do_cut_actions_with_videos, do_pad_stranded_sequencies, play_video, SeededRandomSampler, load_images_as_video_sequence
import matplotlib.pyplot as plt
from ViViT.vivit import ViViT
import seaborn as sns
import time
import math
import os

from PIL import Image


# _________________________________________________________________________________________________________________________
# _________________________________________________________________________________________________________________________
# _________________________________________________________________________________________________________________________


# General variables

path = '/home/s2412003/Shared/JAIST_Cylinder/Synchronized_Dataset_Realtime'
saving_path = '/home/s2412003/Shared/JAIST_Cylinder/Segmented_Dataset_Realtime'

sub_folders = ['IMU', 'Video1', 'Video2']

do_train = True
do_plot = False

# Seed for reproducibility
np.random.seed(0)

# Model parameters
nhead = 8
num_encoder_layers = 2
dim_feedforward = 128

# Initialized later
input_dim = 0
output_dim = 0
max_seq_length = 0

# Training and Evaluation
num_epochs = 200
learning_rate = 0.0001
batch_size = 1

pixel_dim = 224
patch_size = 56
checkpoint_model_name = f'checkpoint_model_video1_{learning_rate}lr_{batch_size}bs_{pixel_dim}px_{patch_size}ps.pt'
confusion_matrix_name = f'confusion_matrix_video1_{learning_rate}lr_{batch_size}bs_{pixel_dim}px_{patch_size}ps.png'

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


imu_files = []
cam1_files = []
cam2_files = []



for trial_dir in os.listdir(path):
    print(trial_dir)

    for hand_dir in os.listdir(os.path.join(path, trial_dir)):
        print(f'\n')
        # print(f'\tTrial: {trial_dir}, Hand: {hand_dir}, Action: {action_dir}')
        temp_path = os.path.join(path, trial_dir, hand_dir)
        print(f'\t{temp_path}')
        npz_container = np.load(os.path.join(temp_path, f'{trial_dir}.npz'))

        # Load the IMU data
        imu_data = npz_container['imu']
        # print(f'\t{imu_data.shape=}')

        # Load image names for camera1
        cam1_data = [os.path.join(temp_path, 'images_1', str(f)+'.jpg') for f in np.sort(npz_container['camera1'])]
        cam2_data = [os.path.join(temp_path, 'images_2', str(f)+'.jpg') for f in np.sort(npz_container['camera2'])]
        # print(f'\t{cam1_filenames.shape=}, {cam2_filenames.shape=}')

        # Remove quaternion data
        imu_data = imu_data[:,:,4:]
        # print(f'\t{imu_data.shape=}')

        # Go from (n_imus, sequence_length, n_features) to (sequence_length, n_imus, n_features)
        imu_data = imu_data.transpose(1, 0, 2)
        # print(f'\t{imu_data.shape=}')

        imu_data = imu_data.reshape(imu_data.shape[0], -1)
        # print(f'\t{imu_data.shape=}')
        imu_files.append(imu_data)
        cam1_files.append(cam1_data)
        cam2_files.append(cam2_data)




print(f'\n{len(imu_files)} IMU files read.')
print(f'{len(cam1_files)} Camera1 files read.')
print(f'{len(cam2_files)} Camera2 files read.')


# # Get max and min values per feature
# max_values = np.max(padded_sequences, axis=(0, 1))
# min_values = np.min(padded_sequences, axis=(0, 1))
# print(f'Maximum values per feature: {max_values}')
# print(f'Minimum values per feature: {min_values}')
# exit()

for i in range(len(imu_files)):
    # print(f'Shape of IMU file {i}: {imu_files[i].shape}')
    # print(f'Shape of IMU file {i}: {imu_files[i].dtype}')
    # print(imu_files[i])
    # Cast to float
    imu_files[i] = imu_files[i].astype(np.float32)
    # print(f'Shape of IMU file {i}: {imu_files[i].shape}')
    # print(f'Shape of IMU file {i}: {imu_files[i].dtype}')
    # print(imu_files[i])

    print(f'Maximum values per feature: {np.max(imu_files[i], axis=0)}')
    print(f'Minimum values per feature: {np.min(imu_files[i], axis=0)}')
    imu_files[i] = (imu_files[i] - np.min(imu_files[i], axis=0)) / (np.max(imu_files[i], axis=0) - np.min(imu_files[i], axis=0))
    print(f'Maximum values per feature: {np.max(imu_files[i], axis=0)}')
    print(f'Minimum values per feature: {np.min(imu_files[i], axis=0)}')
# Load the videos in numpy arrays


print(f'\n{len(imu_files)} IMU files read.')
print(f'{len(cam1_files)} Camera1 files read.')
print(f'{len(cam2_files)} Camera2 files read.')

# video1s = []
# video2s = []

for i in range(len(cam1_files)):
    print(f'Lenghts of {i}-th trial: {len(imu_files[i])}, {len(cam1_files[i])}, {len(cam2_files[i])}')
    assert len(imu_files[i]) == len(cam1_files[i]) == len(cam2_files[i])

    loaded_video1 = load_images_as_video_sequence(cam1_files[i], cam_id=1, pixel_dim=224)
    print(f'Loaded video1: {loaded_video1.shape}')
    # video1s.append(loaded_video1)

    loaded_video2 = load_images_as_video_sequence(cam2_files[i], cam_id=1, pixel_dim=224)
    print(f'Loaded video2: {loaded_video2.shape}')
    # video2s.append(loaded_video2)

    # Check if os.path.join(saving_path, '...') exists
    if not os.path.exists(os.path.join(saving_path, 'IMU')):
        os.makedirs(os.path.join(saving_path, 'IMU'))
    if not os.path.exists(os.path.join(saving_path, 'Video1')):
        os.makedirs(os.path.join(saving_path, 'Video1'))
    if not os.path.exists(os.path.join(saving_path, 'Video2')):
        os.makedirs(os.path.join(saving_path, 'Video2'))

    np.save(os.path.join(saving_path, 'IMU', f'{i}_{imu_files[i].shape[0]}_imu.npy'), imu_files[i])
    np.save(os.path.join(saving_path, 'Video1', f'{i}_{loaded_video2.shape[0]}_video1.npy'), loaded_video1.numpy())
    np.save(os.path.join(saving_path, 'Video2', f'{i}_{loaded_video2.shape[0]}_video2.npy'), loaded_video2.numpy())


print('Finished saving the files.')
exit()
