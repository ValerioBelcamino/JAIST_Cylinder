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
from utils import EarlyStopper, play_video, SeededRandomSampler
import matplotlib.pyplot as plt
from models import BicefHARlo
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

sub_folders = ['Video1', 'Video2']

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
intermediate_dim = 64

# Training and Evaluation
num_epochs = 200
learning_rate = 0.0001
batch_size = 8
patience = 20

video_augmentation = False

pixel_dim = 224
patch_size = 56
max_time = 90
n_features = 72

checkpoint_model_name = f'checkpoint_model_VideoDoubleModel_{learning_rate}lr_{batch_size}bs_{pixel_dim}px_{patch_size}ps_{video_augmentation}Aug.pt'
confusion_matrix_name = f'confusion_matrix_VideoDoubleModel_{learning_rate}lr_{batch_size}bs_{pixel_dim}px_{patch_size}ps_{video_augmentation}Aug.png'

print(f'Saving model to {checkpoint_model_name}')
print(f'Saving confusion matrix to {confusion_matrix_name}')

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


# _________________________________________________________________________________________________________________________
# _________________________________________________________________________________________________________________________
# _________________________________________________________________________________________________________________________

video_filenames1 = []
video_labels1 = []

video_filenames2 = []
video_labels2 = []


print(f'Using both Video1 and Video2...')

for trial in sorted(os.listdir(path)):
    video_names1 = sorted([os.path.join(path, trial, sub_folders[0], f) for f in os.listdir(os.path.join(path, trial, sub_folders[0]))])
    video_filenames1.extend(video_names1)

for trial in sorted(os.listdir(path)):
    video_names2 = sorted([os.path.join(path, trial, sub_folders[1], f) for f in os.listdir(os.path.join(path, trial, sub_folders[1]))])
    video_filenames2.extend(video_names2)



for vn in video_filenames1:
    vn_base = os.path.basename(vn)
    video_labels1.append(int(vn_base.split('_')[1]))

for vn in video_filenames2:
    vn_base = os.path.basename(vn)
    video_labels2.append(int(vn_base.split('_')[1]))


# Convert the label list into a tensor
video_labels1 = torch.tensor(video_labels1)
video_labels2 = torch.tensor(video_labels2)


print(f'{len(video_filenames1)}')
print(f'{len(video_labels1)}')
print(f'{len(video_filenames2)}')
print(f'{len(video_labels2)}')


# Check that there are no duplicates
assert len(video_filenames1) == len(set(video_filenames1)) == len(video_labels1) 
assert len(video_filenames2) == len(set(video_filenames2)) == len(video_labels2)
assert len(video_filenames1) == len(video_filenames2) 


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


Y_train_labels = video_labels1[X_train_idxs]
Y_test_labels = video_labels1[X_test_idxs]



print(f'\n{len(X_train_video1)=}, {len(X_test_video1)=}')
print(f'{len(X_train_video2)=}, {len(X_test_video2)=}')
print(f'{len(Y_train_labels)=}, {len(Y_test_labels)=}')




# Set max time using the dictionary times 30 fps to pad the videos
# max_time = 30*max(action_cut_time_dict.values())

# Create the video datasets
train_dataset_video1 = VideoDatasetNPY(X_train_video1, Y_train_labels, [], video_augmentation, max_length=max_time, pixel_dim=pixel_dim, cam_id=1)
train_dataset_video2 = VideoDatasetNPY(X_train_video2, Y_train_labels, [], video_augmentation, max_length=max_time, pixel_dim=pixel_dim, cam_id=2)
test_dataset_video1 = VideoDatasetNPY( X_test_video1,  Y_test_labels,  [], video_augmentation, max_length=max_time, pixel_dim=pixel_dim, cam_id=1)
test_dataset_video2 = VideoDatasetNPY( X_test_video2,  Y_test_labels,  [], video_augmentation, max_length=max_time, pixel_dim=pixel_dim, cam_id=2)
print('Datasets Initialized\n')

# Create a sampler with a seed to synchronize the dataset loaders
seed = 42
sampler_train = SeededRandomSampler(train_dataset_video1, seed)
sampler_test = SeededRandomSampler(test_dataset_video1, seed)

# Create the training data loaders for the imu and video datasets
train_loader_video1 = DataLoader(train_dataset_video1, batch_size=batch_size, sampler=sampler_train)
train_loader_video2 = DataLoader(train_dataset_video2, batch_size=batch_size, sampler=sampler_train)

# Create the test data loaders for the imu and video datasets
test_loader_video1 = DataLoader( test_dataset_video1,  batch_size=batch_size, sampler=sampler_test)
test_loader_video2 = DataLoader( test_dataset_video2,  batch_size=batch_size, sampler=sampler_test)
print('Data Loaders Initialized\n')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = BicefHARlo(
                      pixel_dim, 
                      patch_size, 
                      len(action_names), 
                      90, 
                      intermediate_dim
                      ).cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
print(f'Model initialized on {device}\n')



# Initialize early stopping
early_stopping = EarlyStopper(saving_path=os.path.join('_new_video_results', checkpoint_model_name), patience=patience)

best_model = None
train_losses = []

print('\n\n\n')

if do_train:
    for epoch in range(num_epochs):
        model.train()
        for (videos1, labels), (videos2, _) in zip(train_loader_video1, train_loader_video2):

            videos1, videos2 = videos1.to(device), videos2.to(device)
            labels = labels.to(device)

            outputs = model(videos1, videos2)
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
model.load_state_dict(torch.load(os.path.join('_new_video_results', checkpoint_model_name)))

# Evaluation
model.eval()
y_true = []
y_pred = []

with torch.no_grad():
    for (videos1, labels), (videos2, _) in zip(test_loader_video1, test_loader_video2):
        
        videos1, videos2 = videos1.to(device), videos2.to(device)
        labels = labels.to(device)

        outputs = model(videos1, videos2)
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
plt.savefig(os.path.join('_new_video_results', confusion_matrix_name))
print(f'Confusion matrix saved to {os.path.join("_new_video_results", confusion_matrix_name)}')
plt.show()
exit()

