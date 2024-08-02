import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from video_dataset import VideoDatasetNPY
from sequence_dataset import SequenceDatasetNPY
from utils import EarlyStopper, play_video, SeededRandomSampler, create_confusion_matrix_w_precision_recall
import matplotlib.pyplot as plt
from models import HAR_Transformer
import seaborn as sns
import time
import math
import os



# _________________________________________________________________________________________________________________________
# _________________________________________________________________________________________________________________________
# _________________________________________________________________________________________________________________________
# a = np.load('/home/s2412003/Shared/JAIST_Cylinder/Segmented_Dataset/0/IMU/0_6_120_imu.npy')
# print(a.shape)
# General variables

path = '/home/s2412003/Shared/JAIST_Cylinder/Segmented_Dataset2'

sub_folders = ['IMU']

do_train = True    

# Seed for reproducibility
np.random.seed(0)

# Initialized later
input_dim = 0
output_dim = 0
max_seq_length = 0
nhead = 16
num_encoder_layers = 2
dim_feedforward = 256

# Training and Evaluation
num_epochs = 200
learning_rate = 0.0005
batch_size = 32
patience = 10

video_augmentation = False

pixel_dim = 224
patch_size = 56
max_time = 90

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


# print(f'{maxes.shape=}, {mins.shape=}')
# exit()


# checkpoit_model_name = os.path.join(path, checkpoint_model_name)
# confusion_matrix_name = os.path.join(path, confusion_matrix_name)


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
                        
all_imus = [0, 1, 2, 3, 4, 5, 6, 7]
tips_and_back = [0, 1, 3, 5]
tips_and_wrist = [7, 1, 3, 5]

tips_only = [1, 3, 5]
thumb_index_tips_only = [1, 5]
thumb_index_back = [0, 1, 5]

back_and_wrist = [0, 7]
thumb_and_index = [1, 2, 5, 6]

sensor_conf = all_imus
sensor_conf_name = 'all_imus'
n_features = 8 * (len(sensor_conf) + 1)
print(f'{n_features=}')



# checkpoint_model_name = f'checkpoint_model_IMUdoubleVideo_{sensor_conf_name}_{learning_rate}lr_{batch_size}bs_{pixel_dim}px_{patch_size}ps_{video_augmentation}Aug.pt'
# confusion_matrix_name = f'confusion_matrix_IMUdoubleVideo_{sensor_conf_name}_{learning_rate}lr_{batch_size}bs_{pixel_dim}px_{patch_size}ps_{video_augmentation}Aug.png'

checkpoint_model_name = f'checkpoint_model_IMU_{sensor_conf_name}_{learning_rate}lr_{batch_size}bs_{video_augmentation}Aug.pt'
confusion_matrix_name = f'confusion_matrix_IMU_{sensor_conf_name}_{learning_rate}lr_{batch_size}bs_{video_augmentation}Aug.png'

print(f'Saving model to {checkpoint_model_name}')
print(f'Saving confusion matrix to {confusion_matrix_name}')



# _________________________________________________________________________________________________________________________
# _________________________________________________________________________________________________________________________
# _________________________________________________________________________________________________________________________


imu_filenames = []
imu_labels = []

lengths = []

print(f'Using both Video1 and Video2...')


for trial in sorted(os.listdir(path)):
    imu_names = sorted([os.path.join(path, trial, sub_folders[0], f) for f in os.listdir(os.path.join(path, trial, sub_folders[0]))])
    imu_filenames.extend(imu_names)


for imn in imu_filenames:
    imn_base = os.path.basename(imn)
    imu_labels.append(int(imn_base.split('_')[1]))
    lengths.append(int(imn_base.split('_')[2]))

# Convert the label list into a tensor
imu_labels = torch.tensor(imu_labels)
lengths = torch.tensor(lengths)

print(f'{len(imu_filenames)}')
print(f'{len(imu_labels)}')

# Check that there are no duplicates
assert len(imu_filenames) == len(set(imu_filenames)) == len(imu_labels)


indices = np.arange(len(imu_filenames))

X_train_idxs, X_test_idxs = train_test_split(
                                            indices, 
                                            test_size=0.3, 
                                            random_state=0
                                            )


X_train_imu = [imu_filenames[i] for i in X_train_idxs]
X_test_imu = [imu_filenames[i] for i in X_test_idxs]

Y_train_labels = imu_labels[X_train_idxs]
Y_test_labels = imu_labels[X_test_idxs]

lengths_train = lengths[X_train_idxs]
lengths_test = lengths[X_test_idxs]

print(f'{len(X_train_imu)=}, {len(X_test_imu)=}')
print(f'{len(Y_train_labels)=}, {len(Y_test_labels)=}')
print(f'{len(lengths_train)=}, {len(lengths_test)=}\n')




# Create the imu datasets
train_dataset_imu = SequenceDatasetNPY(X_train_imu, Y_train_labels, lengths_train, max_len=max_time ,IMU_conf=sensor_conf)
test_dataset_imu = SequenceDatasetNPY(X_test_imu, Y_test_labels, lengths_test, max_len=max_time ,IMU_conf=sensor_conf)

# # Set max time using the dictionary times 30 fps to pad the videos
# max_time = 30*max_time
# print(f'{max_time=}')

print('Datasets Initialized\n')


# Create the training data loaders for the imu and video datasets
train_loader_imu = DataLoader(train_dataset_imu, batch_size=batch_size, shuffle=True)

# Create the test data loaders for the imu and video datasets
test_loader_imu = DataLoader( test_dataset_imu,  batch_size=batch_size, shuffle=True)
print('Data Loaders Initialized\n')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = HAR_Transformer(n_features, nhead, num_encoder_layers, dim_feedforward, len(action_names), max_time).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
print(f'Model initialized on {device}\n')

# Initialize early stopping
early_stopping = EarlyStopper(saving_path=os.path.join('_new_imu_results', checkpoint_model_name), patience=patience)

best_model = None
train_losses = []

print('\n\n\n')

if do_train:
    for epoch in range(num_epochs):
        model.train()
        for imu_seqs, labels, batch_lengths in train_loader_imu:

            # # Normalize the imu sequences using the maxes and mins
            # for i in range(imu_seqs.shape[0]):
            #     imu_seqs[i] = (imu_seqs[i] - mins) / (maxes - mins)

            # Standardize the imu sequences
            for i in range(imu_seqs.shape[0]):
                imu_seqs[i] = (imu_seqs[i] - means) / stds


            imu_seqs = imu_seqs.to(device)

            labels = labels.to(device)
            batch_lengths = batch_lengths.to(device)

            outputs = model(imu_seqs, batch_lengths)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Best Loss: {early_stopping.min_validation_loss:.4f}, Counter: {early_stopping.counter}')

        train_losses.append(loss.item())

        if early_stopping.early_stop(loss.item(), model.state_dict()):
            print("Early stopping")

            # Save the best model
            # torch.save(early_stopping.best_model_state_dict, os.path.join('video_results', checkpoint_model_name))
            # print(f'Model saved to {checkpoint_model_name}')

            break   
        else:
            best_model = model.state_dict()
            # torch.save(model.state_dict(), os.path.join('video_results', checkpoint_model_name))

    # Plot train losses
    plt.figure()
    plt.plot(train_losses)
    plt.title("Train Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()


# Load the best model
model.load_state_dict(torch.load(os.path.join('_new_imu_results', checkpoint_model_name)))

# Evaluation
model.eval()
y_true = []
y_pred = []

with torch.no_grad():
    for imu_seqs, labels, batch_lengths in test_loader_imu:

        # # Normalize the imu sequences using the maxes and mins
        # for i in range(imu_seqs.shape[0]):
        #     imu_seqs[i] = (imu_seqs[i] - mins) / (maxes - mins)

        # Standardize the imu sequences
        for i in range(imu_seqs.shape[0]):
            imu_seqs[i] = (imu_seqs[i] - means) / stds
        
        imu_seqs = imu_seqs.to(device)
        labels = labels.to(device)
        batch_lengths = batch_lengths.to(device)

        outputs = model(imu_seqs, batch_lengths)
        _, predicted = torch.max(outputs.data, 1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

accuracy = accuracy_score(y_true, y_pred)
print(f'Test Accuracy: {accuracy:.4f}')

# Confusion matrix with precision and recall
conf_matrix_ext = create_confusion_matrix_w_precision_recall(y_true, y_pred, accuracy)

# Compute F1 score using scikit-learn
f1 = f1_score(y_true, y_pred, average='macro')
print(f'F1 Score: {f1:.4f}')


# Plot extended confusion matrix
plt.figure(figsize=(16, 9))
sns.heatmap(conf_matrix_ext, annot=True, fmt='.2f', cmap='Blues', xticklabels= action_names + ['Recall'], yticklabels= action_names + ['Precision'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix with Precision and Recall')
plt.savefig(os.path.join('_new_imu_results', confusion_matrix_name[:-4] + f'_f1_{f1:.3f}.png'))
plt.show()
