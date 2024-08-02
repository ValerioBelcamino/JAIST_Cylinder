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


maxes = np.array([15861., 11993., 11564., 317., 496., 334., 125., 91., 47., 15892.,
                  14565., 14710., 620., 996., 335., 77., 89., 60., 15803., 13294.,
                  14115., 440., 691., 334., 76., 141., 69., 15994., 14384., 14890.,
                  497., 1088., 297., 79., 90., 5102., 15738., 12880., 12703., 401.,
                  711., 333., 83., 141., 67., 15459., 14029., 15195., 605., 1338.,
                  515., 91., 89., 80., 16095., 16305., 15268., 4853., 917., 399.,
                  12263., 10005., 69., 14986., 9420., 15614., 143., 450., 182., 106.,
                  53., 29.])

mins = np.array([-1.6198e+04, -1.4711e+04, -1.1353e+04, -3.3600e+02, -9.9800e+02, -4.1600e+02,
                    -3.4000e+01, -8.5000e+01, -1.5300e+02, -1.6571e+04, -1.4554e+04, -1.5535e+04,
                    -5.3900e+02, -6.5500e+02, -2.6800e+02, -4.6000e+01, -5.5000e+01, -1.3600e+02,
                    -1.6414e+04, -1.4992e+04, -1.5140e+04, -5.4300e+02, -5.8000e+02, -4.6200e+02,
                    -6.8000e+01, -7.2000e+01, -1.6400e+02, -1.6419e+04, -1.4184e+04, -1.5454e+04,
                    -5.3400e+02, -7.4600e+02, -4.0600e+02, -4.7000e+01, -4.3000e+01, -1.3600e+02,
                    -1.6552e+04, -1.5281e+04, -1.4352e+04, -4.4000e+02, -7.2300e+02, -4.0900e+02,
                    -7.0000e+01, -7.1000e+01, -1.5200e+02, -1.6015e+04, -1.5723e+04, -1.5722e+04,
                    -5.9800e+02, -9.3100e+02, -7.2300e+02, -4.5000e+01, -8.8000e+01, -1.5300e+02,
                    -1.6672e+04, -1.5108e+04, -2.9679e+04, -5.2200e+02, -3.2615e+04, -1.5807e+04,
                    -7.1000e+01, -9.5000e+01, -1.1966e+04, -1.6042e+04, -1.4300e+04, -9.0130e+03,
                    -1.3300e+02, -4.4600e+02, -1.5200e+02, -6.0000e+00, -9.8000e+01, -1.0500e+02])

means = np.array([-6.96805542e+02, -2.25853418e+03, -1.34595679e+03, -1.11699450e+00,
                    5.86712211e-02, -1.07630253e+00, 3.19620800e+01, 2.18175774e+01,
                    -3.43563194e+01, -3.16794373e+02, -1.14215515e+02, -1.60282751e+03,
                    -2.34029472e-01, -5.03993154e-01, -7.03439653e-01, 7.47513914e+00,
                    2.69293461e+01, -3.57804832e+01, -3.46590729e+02, -1.82277844e+03,
                    -1.72432056e+03, -1.27731073e+00, -2.57041603e-01, -3.78725111e-01,
                    9.31902599e+00, 2.67350693e+01, -3.00858269e+01, -2.16260269e+02,
                    2.93784943e+02, -5.63845581e+02, -2.28240028e-01, -2.25193590e-01,
                    -6.19725347e-01, 7.64884233e+00, 2.62607403e+01, -3.29219246e+01,
                    -3.30930664e+02, -1.08913989e+03, -1.42398950e+03, -9.96334553e-01,
                    -9.73807927e-03, -2.20419630e-01, 2.81873083e+00, 2.35137424e+01,
                    -3.13092594e+01, 2.39130508e+02, -1.68099585e+03, -2.52542114e+03,
                    -1.48842512e-02, -1.63395017e-01, -4.47753400e-01, 1.08532734e+01,
                    2.09372749e+01, -4.61218376e+01, 1.10423950e+03, -1.25824939e+03,
                    -3.75632471e+03, -1.44030523e+00, 4.82494123e-02, 7.91183561e-02,
                    5.89785719e+00, 2.82478294e+01, -7.24016800e+01, -9.28175781e+02,
                    9.93521667e+02, 3.95115112e+03, -3.51719826e-01, -5.98796785e-01,
                    -2.33891919e-01, 5.38646889e+01, -3.33873701e+00, -2.15550632e+01])


stds = np.array([6945.848, 2686.427, 2871.097, 32.93815, 37.17426, 27.305304,
                    17.970592, 44.100662, 27.643883, 7298.917, 2551.3857, 3329.0312,
                    51.304676, 49.43646, 31.335526, 16.893368, 47.71109, 32.661713,
                    6938.176, 2715.3987, 3296.7, 44.91464, 41.5619, 31.240667,
                    22.5143, 53.592587, 46.943756, 7372.4243, 2737.9797, 3452.1938,
                    49.012985, 50.870693, 31.164236, 19.169243, 41.900204, 35.39268,
                    7077.101, 2801.966, 3288.3171, 43.516663, 43.00801, 30.701008,
                    23.039852, 70.481316, 48.276485, 5794.938, 3421.688, 4203.817,
                    44.632915, 45.571747, 34.52821, 23.115152, 35.290333, 37.143223,
                    6662.959, 4907.151, 4953.2593, 33.83719, 76.69994, 47.600132,
                    38.09727, 62.66252, 63.31026, 6549.3613, 2293.787, 2729.528,
                    15.296869, 30.679577, 17.03959, 15.159824, 43.69038, 22.054604])

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
    # imu_files[i] = (imu_files[i] - mins) / (maxes - mins)
    imu_files[i] = (imu_files[i] - means) / stds
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
