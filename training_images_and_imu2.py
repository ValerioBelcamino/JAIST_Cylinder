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
from utils import EarlyStopper, do_cut_actions_with_videos, do_pad_stranded_sequencies, play_video, SeededRandomSampler
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

path = '/home/s2412003/Shared/JAIST_Cylinder/Segmented_Dataset'

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

video1_filenames = []
video2_filenames = []
imu_filenames = []

trials = os.listdir(path)
for trial in sorted(trials):
    # imu_names = sorted([f'{trial}_{f}' for f in os.listdir(os.path.join(path, trial, sub_folders[0]))])
    # video_names1 = sorted([f'{trial}_{f}' for f in os.listdir(os.path.join(path, trial, sub_folders[1]))])
    # video_names2 = sorted([f'{trial}_{f}' for f in os.listdir(os.path.join(path, trial, sub_folders[2]))])
    imu_names = sorted([os.path.join(path, trial, sub_folders[0], f) for f in os.listdir(os.path.join(path, trial, sub_folders[0]))])
    video_names1 = sorted([os.path.join(path, trial, sub_folders[1], f) for f in os.listdir(os.path.join(path, trial, sub_folders[1]))])
    video_names2 = sorted([os.path.join(path, trial, sub_folders[2], f) for f in os.listdir(os.path.join(path, trial, sub_folders[2]))])

    imu_filenames.extend(imu_names)
    video1_filenames.extend(video_names1)
    video2_filenames.extend(video_names2)



print(f'\n{video1_filenames[:10]}')
print(f'{len(video1_filenames)}')

print(f'\n{video2_filenames[:10]}')
print(f'{len(video2_filenames)}')

print(f'\n{imu_filenames[:10]}')
print(f'{len(imu_filenames)}')

# Check that they are the same and no duplicates
assert len(video1_filenames) == len(video2_filenames) == len(imu_filenames)
assert len(video1_filenames) == len(set(video1_filenames)) == len(set(video2_filenames)) == len(set(imu_filenames)) 

for i in range(len(video1_filenames)):
    tmp_vid1_name = '_'.join(os.path.basename(video1_filenames[i]).split('_')[:-1])
    tmp_vid2_name = '_'.join(os.path.basename(video2_filenames[i]).split('_')[:-1])
    tmp_imu_name = '_'.join(os.path.basename(imu_filenames[i]).split('_')[:-1])

    tmp_vid1_dir = '/'.join(os.path.dirname(video1_filenames[i]).split('/')[:-1])
    tmp_vid2_dir = '/'.join(os.path.dirname(video2_filenames[i]).split('/')[:-1])
    tmp_imu_dir = '/'.join(os.path.dirname(imu_filenames[i]).split('/')[:-1])

    print(f'\n{tmp_vid1_dir}')
    print(f'{tmp_vid1_name=}, {tmp_vid2_name=}', f'{tmp_imu_name=}')
    assert tmp_vid1_name == tmp_vid2_name == tmp_imu_name
    assert tmp_vid1_dir == tmp_vid2_dir == tmp_imu_dir

# cam1_files_names = os.listdir(os.path.join(path, sub_folders[1]))
start_time = time.time()
for i in range(len(video1_filenames)):
    np.load(video1_filenames[i])
print(f'\n{time.time() - start_time}')

exit()

imu_files = []
cam1_files = []
cam2_files = []
imu_labels = []

for trial_dir in os.listdir(path):
    print(trial_dir)
    if trial_dir != '0':
        continue
    for hand_dir in os.listdir(os.path.join(path, trial_dir)):
        print(f'\n')
        for action_dir in os.listdir(os.path.join(path, trial_dir, hand_dir)):
            # print(f'\tTrial: {trial_dir}, Hand: {hand_dir}, Action: {action_dir}')
            temp_path = os.path.join(path, trial_dir, hand_dir, action_dir)
            print(f'\t{temp_path}')
            npz_container = np.load(os.path.join(temp_path, f'{action_dir}.npz'))

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
            imu_labels.append(action_dir)
            cam1_files.append(cam1_data)
            cam2_files.append(cam2_data)




print(f'\n{len(imu_files)} IMU files read.')
print(f'{len(imu_labels)} labels created.')
print(f'{len(cam1_files)} Camera1 files read.')
print(f'{len(cam2_files)} Camera2 files read.')


imu_sequencies, cam1_sequencies, cam2_sequencies, imu_labels, sequence_lengths = do_cut_actions_with_videos(imu_files, cam1_files, cam2_files, imu_labels, action_cut_time_dict)

print(f'\n{len(imu_sequencies)} sequences created.'
      f'\n{len(imu_labels)} labels created.'
      f'\n{len(sequence_lengths)} sequence lengths created.'
      f'\n{len(cam1_sequencies)} Camera1 sequences created.'
        f'\n{len(cam2_sequencies)} Camera2 sequences created.')

# print(f'\n{imu_sequencies[0].shape=}, {cam1_sequencies[0].shape=}, {cam2_sequencies[0].shape=}')
# print(f'{cam1_sequencies[0][0]}')
# print(f'{len(cam2_sequencies[0])}')

# Image.open(cam1_sequencies[0][0]).show()
# exit()

# Convert labels to numerical values
label_indices = [action_dict[label] for label in imu_labels]

# Calculate the lengths of the sequences
lengths = [data.shape[0] / 30 for data in imu_sequencies]

# Create a dictionary to store the lengths for each label
lengths_dict = {}
for label, length in zip(label_indices, lengths):
    if label not in lengths_dict:
        lengths_dict[label] = []
    lengths_dict[label].append(length)

# Create a list of lengths for each label
lengths_per_label = [lengths_dict[label] for label in range(len(action_names))]

print('\n')
if do_plot:
    # Create a boxplot for each label
    plt.boxplot(lengths_per_label, labels=action_names)
    plt.title("Sequence Lengths in seconds")
    plt.xlabel("Action Labels")
    plt.ylabel("Length")
    plt.ylim(0, 6)
    plt.show()

print('\n')
# Calculate the average length of each label
average_lengths = []
for label in range(len(action_names)):
    lengths = lengths_dict[label]
    average_length = sum(lengths) / len(lengths)
    average_lengths.append(average_length)
    print(f"Average length for label '{action_dict_inv[label]}': {average_length:.2f} seconds")

print('\n')
padded_sequences = do_pad_stranded_sequencies(imu_sequencies)
print(f'The shape of the padded sequences is {padded_sequences.shape}.')
assert padded_sequences.shape[0] == len(label_indices) == len(sequence_lengths)

# Get max and min values per feature
max_values = np.max(padded_sequences, axis=(0, 1))
min_values = np.min(padded_sequences, axis=(0, 1))
# print(f'Maximum values per feature: {max_values}')

padded_sequences = (padded_sequences - min_values) / (max_values - min_values)

indices = np.arange(len(label_indices))

# X_train_indices, X_test_indices, y_train, y_test, lengths_train, lengths_test = train_test_split(indices,
#                                                                                  label_indices, 
#                                                                                  sequence_lengths, 
#                                                                                  test_size=0.3, 
#                                                                                  random_state=0)

X_train_idxs, X_test_idxs = train_test_split(
                                            indices, 
                                            test_size=0.3, 
                                            random_state=0
                                            )

print(f'\n{X_train_idxs[:10]=}, {X_test_idxs[:10]=}'
      f'\n{len(X_train_idxs)=}, {len(X_test_idxs)=}')

# Concat train and test
X_train_idxs = np.concatenate([X_train_idxs, X_test_idxs])
print(f'\n{len(X_train_idxs)=}')
# Check no duplicates
assert len(set(X_train_idxs)) == len(X_train_idxs)
print(len(set(X_train_idxs)) == len(X_train_idxs))

X_train_imu = padded_sequences[X_train_idxs]
X_test_imu = padded_sequences[X_test_idxs]

X_train_video1 = [cam1_sequencies[i] for i in X_train_idxs]
X_test_video1 = [cam1_sequencies[i] for i in X_test_idxs]

X_train_video2 = [cam2_sequencies[i] for i in X_train_idxs]
X_test_video2 = [cam2_sequencies[i] for i in X_test_idxs]

y_train = [label_indices[i] for i in X_train_idxs]
y_test = [label_indices[i] for i in X_test_idxs]

lengths_train = [sequence_lengths[i] for i in X_train_idxs]
lengths_test = [sequence_lengths[i] for i in X_test_idxs]

# print(f'\n{X_train_indices[:10]=}, {X_test_indices[:10]=}')
# X_train = padded_sequences[X_train_indices]
# X_test = padded_sequences[X_test_indices]
print('\n')
print(f'{X_train_imu.shape=}, {X_test_imu.shape=}')
print(f'{len(X_train_video1)=}, {len(X_test_video1)=}')
print(f'{len(X_train_video2)=}, {len(X_test_video2)=}')
print(f'{len(y_train)=}, {len(y_test)=}')
print(f'{len(lengths_train)=}, {len(lengths_test)=}')

# Convert to PyTorch tensors
X_train_imu = torch.tensor(X_train_imu, dtype=torch.float32)
y_train = torch.tensor(y_train)
lengths_train = torch.tensor(lengths_train)
X_test_imu = torch.tensor(X_test_imu, dtype=torch.float32)
y_test = torch.tensor(y_test)
lengths_test = torch.tensor(lengths_test)

print(f'\n{X_train_imu.shape=}, {y_train.shape=}, {lengths_train.shape=}, {X_train_imu.dtype=}')
print(f'{X_test_imu.shape=}, {y_test.shape=}, {lengths_test.shape=}, {X_test_imu.dtype=}')



# Create the imu datasets
train_dataset_imu = SequenceDataset(X_train_imu, y_train, lengths_train)
test_dataset_imu = SequenceDataset(X_test_imu, y_test, lengths_test)

# Set max time using the dictionary times 30 fps to pad the videos
max_time = 30*max(action_cut_time_dict.values())

# Create the video datasets
train_dataset_video1 = VideoDataset(X_train_video1, y_train, lengths_train, max_length=max_time, pixel_dim=pixel_dim)
train_dataset_video2 = VideoDataset(X_train_video2, y_train, lengths_train, max_length=max_time, pixel_dim=pixel_dim)
test_dataset_video1 = VideoDataset( X_test_video1,  y_test,  lengths_train, max_length=max_time, pixel_dim=pixel_dim)
test_dataset_video2 = VideoDataset( X_test_video2,  y_test,  lengths_train, max_length=max_time, pixel_dim=pixel_dim)


# Create a sampler with a seed to synchronize the dataset loaders
seed = 42
sampler_train = SeededRandomSampler(train_dataset_imu, seed)
sampler_test = SeededRandomSampler(test_dataset_imu, seed)

# Create the training data loaders for the imu and video datasets
train_loader_imu = DataLoader(train_dataset_imu, batch_size=batch_size, sampler=sampler_train)
train_loader_video1 = DataLoader(train_dataset_video1, batch_size=batch_size, sampler=sampler_train)
train_loader_video2 = DataLoader(train_dataset_video2, batch_size=batch_size, sampler=sampler_train)

# Create the test data loaders for the imu and video datasets
test_loader_imu = DataLoader( test_dataset_imu,  batch_size=batch_size, sampler=sampler_test)
test_loader_video1 = DataLoader( test_dataset_video1,  batch_size=batch_size, sampler=sampler_test)
test_loader_video2 = DataLoader( test_dataset_video2,  batch_size=batch_size, sampler=sampler_test)



''' Check if the data loaders are working '''
''' Check that they don't have duplicates '''

# # _________________________________________________________________________________________________________________________
# # _________________________________________________________________________________________________________________________
# # _________________________________________________________________________________________________________________________


# imu_train_loader_idxs = [idx for idx in train_loader_imu]
# # print(imu_train_loader_idxs)
# imu_train_loader_idxs = torch.cat(imu_train_loader_idxs).tolist()
# imu_train_loader_idxs_set = set(imu_train_loader_idxs)

# video1_train_loader_idxs = [idx for idx in train_loader_video1]
# video1_train_loader_idxs = torch.cat(video1_train_loader_idxs).tolist()
# video1_train_loader_idxs_set = set(video1_train_loader_idxs)

# video2_train_loader_idxs = [idx for idx in train_loader_video2]
# video2_train_loader_idxs = torch.cat(video2_train_loader_idxs).tolist()
# video2_train_loader_idxs_set = set(video2_train_loader_idxs)

# # Check that the indices are the same for all data loaders and same order
# assert imu_train_loader_idxs == video1_train_loader_idxs == video2_train_loader_idxs
# assert imu_train_loader_idxs_set == video1_train_loader_idxs_set == video2_train_loader_idxs_set
# assert len(imu_train_loader_idxs) == len(imu_train_loader_idxs_set)
# assert len(video1_train_loader_idxs) == len(video1_train_loader_idxs_set)
# assert len(video2_train_loader_idxs) == len(video2_train_loader_idxs_set)

# imu_test_loader_idxs = [idx for idx in test_loader_imu]
# imu_test_loader_idxs = torch.cat(imu_test_loader_idxs).tolist()
# imu_test_loader_idxs_set = set(imu_test_loader_idxs)

# video1_test_loader_idxs = [idx for idx in test_loader_video1]
# video1_test_loader_idxs = torch.cat(video1_test_loader_idxs).tolist()
# video1_test_loader_idxs_set = set(video1_test_loader_idxs)

# video2_test_loader_idxs = [idx for idx in test_loader_video2]
# video2_test_loader_idxs = torch.cat(video2_test_loader_idxs).tolist()
# video2_test_loader_idxs_set = set(video2_test_loader_idxs)

# # Check that the indices are the same for all data loaders and same order
# assert imu_test_loader_idxs == video1_test_loader_idxs == video2_test_loader_idxs
# assert imu_test_loader_idxs_set == video1_test_loader_idxs_set == video2_test_loader_idxs_set
# assert len(imu_test_loader_idxs) == len(imu_test_loader_idxs_set)
# assert len(video1_test_loader_idxs) == len(video1_test_loader_idxs_set)
# assert len(video2_test_loader_idxs) == len(video2_test_loader_idxs_set)

# print(len(video2_train_loader_idxs))
# print(len(video2_test_loader_idxs))
# exit()

# # _________________________________________________________________________________________________________________________
# # _________________________________________________________________________________________________________________________
# # _________________________________________________________________________________________________________________________


# # ''' Test if you can play the videos '''
# _, seq, lab = next(iter(train_loader_video1))
# # for i in range(seq.shape[0]):
# #     print(f'{i}: {seq[i].shape=}, {lab[i].item()=}')
# #     play_video(seq[i])

# print(seq.shape)
# print(seq.numpy().shape)
print('\n\n\n')
count = 0
for (imu_idx, imu_seq, imu_lab, seq_len), (vid1_idx, vid1_seq, vid1_lab), (vid2_idx, vid2_seq, vid2_lab) in zip(train_loader_imu, train_loader_video1, train_loader_video2):
    count += 1
    total = len(train_loader_imu)
    print(f'{count}/{total} - {imu_idx=}, {imu_seq.shape=}, {imu_lab=}, {seq_len=}')
    # print(f'{imu_idx=}, {imu_seq.shape=}, {imu_lab=}, {seq_len=}')
    # print(f'{vid1_idx=}, {vid1_seq.shape=}, {vid1_lab=}')
    # print(f'{vid2_idx=}, {vid2_seq.shape=}, {vid2_lab=}')

    for i in range(imu_idx.shape[0]):
        assert imu_idx[i] == vid1_idx[i] == vid2_idx[i]
        assert imu_lab[i] == vid1_lab[i] == vid2_lab[i]
        # print(imu_idx[i] == vid1_idx[i] == visd2_idx[i])

    np.save(os.path.join(saving_path, 'IMU', f'{imu_idx[0]}_imu_{imu_lab[0]}_{seq_len.item()}_imu.npy'), imu_seq[0].numpy())
    np.save(os.path.join(saving_path, 'Video1', f'{vid1_idx[0]}_video1_{vid1_lab[0]}_{seq_len.item()}_video1.npy'), vid1_seq[0].numpy())
    np.save(os.path.join(saving_path, 'Video2', f'{vid2_idx[0]}_video2_{vid2_lab[0]}_{seq_len.item()}_video2.npy'), vid2_seq[0].numpy())
    # print(f'{imu_seq[0].shape=}')
    # print(f'{vid1_seq[0].shape=}')
    # print(f'{vid2_seq[0].shape=}')
    # print(f'{imu_seq[0].dtype=}')
    # print(f'{vid1_seq[0].dtype=}')
    # print(f'{vid2_seq[0].dtype=}')
    # break
exit()

# Reinitialize some parameters for the model
input_dim = padded_sequences.shape[2]
output_dim = len(action_names)
max_seq_length = padded_sequences.shape[1]

model = HAR_Transformer(input_dim, nhead, num_encoder_layers, dim_feedforward, output_dim, max_seq_length)
model = ViViT(pixel_dim, patch_size, len(action_names), 150, in_channels=1).cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Initialize early stopping
early_stopping = EarlyStopper(patience=10)

best_model = None
train_losses = []

# previus_time = 0
# for i, (features, labels) in enumerate(train_loader_video1):
#     current_time = time.time()
#     print(f'{i=}, {current_time - previus_time=}')
#     previus_time = current_time

#     new_start = time.time()
#     features, labels = features.to(device), labels.to(device)
#     outputs = model(features)
#     loss = criterion(outputs, labels)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#     new_current_time = time.time()
#     print(f'model time {new_current_time - new_start=}')
#     new_start = new_current_time

#     # print(i)
#     continue

# exit()

# Training loop
if do_train:
    for epoch in range(num_epochs):
        model.train()
        for i, (features, labels) in enumerate(train_loader_video1):
            features, labels = features.to(device), labels.to(device)
            # print(f'{features.shape=}, {labels.shape=}')
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Best Loss: {early_stopping.min_validation_loss:.4f}, Counter: {early_stopping.counter}')

        train_losses.append(loss.item())

        if early_stopping.early_stop(loss.item()):
            print("Early stopping")

            # Save the best model
            torch.save(model.state_dict(), checkpoint_model_name)

            # break
        else:
            best_model = model.state_dict()
            torch.save(model.state_dict(), checkpoint_model_name)

    # Plot train losses
    plt.figure()
    plt.plot(train_losses)
    plt.title("Train Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()


# Load the best model
model.load_state_dict(torch.load(checkpoint_model_name))

# Evaluation
model.eval()
y_true = []
y_pred = []

with torch.no_grad():
    for features, labels in test_loader_video1:
        features, labels = features.to(device), labels.to(device)
        outputs = model(features)
        _, predicted = torch.max(outputs.data, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

accuracy = accuracy_score(y_true, y_pred)
print(f'Test Accuracy: {accuracy:.4f}')

# Confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)

# Calculate precision and recall for each class
precision = precision_score(y_true, y_pred, average=None)
recall = recall_score(y_true, y_pred, average=None)

# From the test_labels, create a dictionary with the number of samples per class
test_labels_dict = {}
for label in y_true:
    if label not in test_labels_dict:
        test_labels_dict[label] = 0
    test_labels_dict[label] += 1

# print(f'\n{test_labels_dict=}')

# Normalize the confusion matrix by the number of samples per class
# print(f'\n{conf_matrix=}')
for i in range(conf_matrix.shape[0]):
    # print(f'{conf_matrix[i, :]}, {test_labels_dict[i]}\n {conf_matrix[i, :] / test_labels_dict[i]}')
    conf_matrix[i, :] = conf_matrix[i, :] / test_labels_dict[i] * 100
# print(f'\n{conf_matrix=}')

# Append precision and recall to the confusion matrix
recall = recall * 100
precision = precision * 100
accuracy = accuracy * 100

conf_matrix_ext = np.c_[conf_matrix, precision]
recall_ext = np.append(recall, accuracy)  # Add a nan for the last cell in recall row
conf_matrix_ext = np.vstack([conf_matrix_ext, recall_ext])


# Plot extended confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix_ext, annot=True, fmt='.2f', cmap='Blues', xticklabels= action_names + ['Recall'], yticklabels= action_names + ['Precision'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix with Precision and Recall')
plt.savefig(confusion_matrix_name)
plt.show()