import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from video_dataset import VideoDatasetNPY
from utils import EarlyStopper, play_video, create_confusion_matrix_w_precision_recall
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

path = '/home/s2412003/Shared/JAIST_Cylinder/Segmented_Dataset1'

sub_folders = ['Video1', 'Video2']

which_camera = 0

do_train = True

# Seed for reproducibility
np.random.seed(0)

# Initialized later
input_dim = 0
output_dim = 0
max_seq_length = 0

# Training and Evaluation
num_epochs = 300
learning_rate = 0.0001
batch_size = 8
patience = 40

video_augmentation = True

pixel_dim = 224
patch_size = 56
max_time = 90

checkpoint_model_name = f'checkpoint_model_{sub_folders[which_camera]}_{learning_rate}lr_{batch_size}bs_{pixel_dim}px_{patch_size}ps_{video_augmentation}aug.pt'
confusion_matrix_name = f'confusion_matrix_{sub_folders[which_camera]}_{learning_rate}lr_{batch_size}bs_{pixel_dim}px_{patch_size}ps_{video_augmentation}aug.png'

print(f'Saving model to {checkpoint_model_name}')
print(f'Saving confusion matrix to {confusion_matrix_name}')

# checkpoit_model_name = os.path.join(path, checkpoint_model_name)
# confusion_matrix_name = os.path.join(path, confusion_matrix_name)


action_names = ['linger', 'massaging', 'patting', 
                'pinching', 'press', 'pull', 
                'push', 'rub', 'scratching', 
                'shaking', 'squeeze', 'stroke', 
                'tapping', 'trembling', 'idle']

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

for vn in video_filenames:
    vn_base = os.path.basename(vn)
    video_labels.append(int(vn_base.split('_')[1]))

# Convert the label list into a tensor
video_labels = torch.tensor(video_labels)

# for i in range(10):
#     print(f'{video_filenames[i]}')
#     print(f'{video_labels[i]}')
# print(f'\n{video_filenames[:10]}')
print(f'{len(video_filenames)}')
print(f'{len(video_labels)}')

# Check that there are no duplicates
assert len(video_filenames) == len(set(video_filenames)) == len(video_labels)


# start_time = time.time()
# for i in range(len(video_filenames)):
#     np.load(video_filenames[i])
# print(f'\n{time.time() - start_time}')

X_train_names, X_test_names, Y_train_labels, Y_test_labels = train_test_split(
                                            video_filenames, 
                                            video_labels,
                                            test_size=0.3, 
                                            random_state=0
                                            )

print(f'\n{len(X_train_names)=}, {len(X_test_names)=}')
print(f'{len(Y_train_labels)=}, {len(Y_test_labels)=}\n')

train_dataset = VideoDatasetNPY(X_train_names, Y_train_labels, [], video_augmentation, max_length=max_time, pixel_dim=pixel_dim, cam_id=1)
test_dataset = VideoDatasetNPY( X_test_names,  Y_test_labels,  [], video_augmentation, max_length=max_time, pixel_dim=pixel_dim, cam_id=2)
print('Datasets Initialized\n')

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print('Data Loaders Initialized\n')

model = ViViT(pixel_dim, patch_size, len(action_names), 90, in_channels=1).cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
print('Model Initialized\n')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(f'Model moved to {device}\n')

# Initialize early stopping
early_stopping = EarlyStopper(saving_path=os.path.join('_new_video_results', checkpoint_model_name), patience=patience)

best_model = None
train_losses = []

print('\n\n\n')

if do_train:
    for epoch in range(num_epochs):
        model.train()
        for videos, labels in train_loader:
            videos, labels = videos.to(device), labels.to(device)

            outputs = model(videos)
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
    for videos, labels in test_loader:
        videos, labels = videos.to(device), labels.to(device)
        outputs = model(videos)
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
plt.savefig(os.path.join('_new_video_results', confusion_matrix_name.split('.')[0] + f'_f1_{f1:.4f}.png'))
plt.show()


