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
from models import CHARberoViVit
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

path = '/home/s2412003/Shared/JAIST_Cylinder/Segmented_Dataset1'

sub_folders = ['Video1', 'Video2', 'IMU']

do_train = True    

# Seed for reproducibility
np.random.seed(0)

# Initialized later
input_dim = 0
output_dim = 0
max_seq_length = 0
nhead = 8  
num_encoder_layers = 2
dim_feedforward = 128
intermediate_dim = 64

# Training and Evaluation
num_epochs = 200
learning_rate = 0.0001
batch_size = 32
patience = 10

video_augmentation = False

pixel_dim = 224
patch_size = 56
max_time = 150


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

checkpoint_model_name = f'checkpoint_model_IMUdoubleVideo_{learning_rate}lr_{batch_size}bs_{pixel_dim}px_{patch_size}ps_{video_augmentation}Aug.pt'
confusion_matrix_name = f'confusion_matrix_IMUdoubleVideo_{learning_rate}lr_{batch_size}bs_{pixel_dim}px_{patch_size}ps_{video_augmentation}Aug.png'

print(f'Saving model to {checkpoint_model_name}')
print(f'Saving confusion matrix to {confusion_matrix_name}')



# _________________________________________________________________________________________________________________________
# _________________________________________________________________________________________________________________________
# _________________________________________________________________________________________________________________________

video_filenames1 = []
video_labels1 = []

video_filenames2 = []
video_labels2 = []

imu_filenames = []
imu_labels = []

lengths = []

print(f'Using both Video1 and Video2...')

for trial in sorted(os.listdir(path)):
    video_names1 = sorted([os.path.join(path, trial, sub_folders[0], f) for f in os.listdir(os.path.join(path, trial, sub_folders[0]))])
    video_filenames1.extend(video_names1)

for trial in sorted(os.listdir(path)):
    video_names2 = sorted([os.path.join(path, trial, sub_folders[1], f) for f in os.listdir(os.path.join(path, trial, sub_folders[1]))])
    video_filenames2.extend(video_names2)

for trial in sorted(os.listdir(path)):
    imu_names = sorted([os.path.join(path, trial, sub_folders[2], f) for f in os.listdir(os.path.join(path, trial, sub_folders[2]))])
    imu_filenames.extend(imu_names)

for vn in video_filenames1:
    vn_base = os.path.basename(vn)
    # print(f'{vn_base=}')
    video_labels1.append(int(vn_base.split('_')[2]))
    lengths.append(int(vn_base.split('_')[3]))

for vn in video_filenames2:
    vn_base = os.path.basename(vn)
    video_labels2.append(int(vn_base.split('_')[2]))

for imn in imu_filenames:
    imn_base = os.path.basename(imn)
    imu_labels.append(int(imn_base.split('_')[2]))

# Convert the label list into a tensor
video_labels1 = torch.tensor(video_labels1)
video_labels2 = torch.tensor(video_labels2)
imu_labels = torch.tensor(imu_labels)
lengths = torch.tensor(lengths)

# for i in range(10):
#     print(f'{video_filenames[i]}')
#     print(f'{video_labels[i]}')
# print(f'\n{video_filenames[:10]}')
print(f'{len(video_filenames1)}')
print(f'{len(video_labels1)}')
print(f'{len(video_filenames2)}')
print(f'{len(video_labels2)}')
print(f'{len(imu_filenames)}')
print(f'{len(imu_labels)}')

# Check that there are no duplicates
assert len(video_filenames1) == len(set(video_filenames1)) == len(video_labels1) 
assert len(video_filenames2) == len(set(video_filenames2)) == len(video_labels2)
assert len(imu_filenames) == len(set(imu_filenames)) == len(imu_labels)
assert len(video_filenames1) == len(video_filenames2) == len(imu_filenames)


indices = np.arange(len(video_filenames1))

X_train_idxs, X_test_idxs = train_test_split(
                                            indices, 
                                            test_size=0.3, 
                                            random_state=0
                                            )

X_train_video1 = [video_filenames1[i] for i in X_train_idxs]
X_test_video1 = [video_filenames1[i] for i in X_test_idxs]

X_train_video2 = [video_filenames2[i] for i in X_train_idxs]
X_test_video2 = [video_filenames2[i] for i in X_test_idxs]

X_train_imu = [imu_filenames[i] for i in X_train_idxs]
X_test_imu = [imu_filenames[i] for i in X_test_idxs]

Y_train_labels = video_labels1[X_train_idxs]
Y_test_labels = video_labels1[X_test_idxs]

lengths_train = lengths[X_train_idxs]
lengths_test = lengths[X_test_idxs]

print(f'\n{len(X_train_video1)=}, {len(X_test_video1)=}')
print(f'{len(X_train_video2)=}, {len(X_test_video2)=}')
print(f'{len(X_train_imu)=}, {len(X_test_imu)=}')
print(f'{len(Y_train_labels)=}, {len(Y_test_labels)=}')
print(f'{len(lengths_train)=}, {len(lengths_test)=}\n')




# Create the imu datasets
train_dataset_imu = SequenceDatasetNPY(X_train_imu, Y_train_labels, lengths_train, IMU_conf=sensor_conf)
test_dataset_imu = SequenceDatasetNPY(X_test_imu, Y_test_labels, lengths_test, IMU_conf=sensor_conf)

# Set max time using the dictionary times 30 fps to pad the videos
max_time = 30*max(action_cut_time_dict.values())

# Create the video datasets
train_dataset_video1 = VideoDatasetNPY(X_train_video1, Y_train_labels, lengths_train, video_augmentation, max_length=max_time, pixel_dim=pixel_dim)
train_dataset_video2 = VideoDatasetNPY(X_train_video2, Y_train_labels, lengths_train, video_augmentation, max_length=max_time, pixel_dim=pixel_dim)
test_dataset_video1 = VideoDatasetNPY( X_test_video1,  Y_test_labels,  lengths_train, video_augmentation, max_length=max_time, pixel_dim=pixel_dim)
test_dataset_video2 = VideoDatasetNPY( X_test_video2,  Y_test_labels,  lengths_train, video_augmentation, max_length=max_time, pixel_dim=pixel_dim)
print('Datasets Initialized\n')

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
print('Data Loaders Initialized\n')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = CHARberoViVit(
                      pixel_dim, 
                      patch_size, 
                      len(action_names), 
                      150, 
                      n_features, 
                      nhead, 
                      num_encoder_layers, 
                      dim_feedforward, 
                      intermediate_dim
                      ).cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
print(f'Model initialized on {device}\n')


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


# Initialize early stopping
early_stopping = EarlyStopper(saving_path=os.path.join('video_imu_results', checkpoint_model_name), patience=patience)

best_model = None
train_losses = []

print('\n\n\n')

if do_train:
    for epoch in range(num_epochs):
        model.train()
        for (videos1, labels), (videos2, _), (imu_seqs, _, batch_lengths) in zip(train_loader_video1, train_loader_video2, train_loader_imu):

            videos1, videos2 = videos1.to(device), videos2.to(device)
            imu_seqs = imu_seqs.to(device)
            labels = labels.to(device)
            batch_lengths = batch_lengths.to(device)

            outputs = model(videos1, videos2, imu_seqs, batch_lengths)
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
model.load_state_dict(torch.load(os.path.join('video_imu_results', checkpoint_model_name)))

# Evaluation
model.eval()
y_true = []
y_pred = []

with torch.no_grad():
    for (videos1, labels), (videos2, _), (imu_seqs, _, batch_lengths) in zip(test_loader_video1, test_loader_video2, test_loader_imu):
        
        videos1, videos2 = videos1.to(device), videos2.to(device)
        imu_seqs = imu_seqs.to(device)
        labels = labels.to(device)
        batch_lengths = batch_lengths.to(device)

        outputs = model(videos1, videos2, imu_seqs, batch_lengths)
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
plt.savefig(os.path.join('video_imu_results', confusion_matrix_name[:-4] + f'_f1_{f1:.3f}.png'))
plt.show()
