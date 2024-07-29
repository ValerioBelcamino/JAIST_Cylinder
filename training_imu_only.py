import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from models import HAR_Transformer
from sequence_dataset import SequenceDataset
from utils import EarlyStopper, do_cut_actions, do_pad_stranded_sequencies, create_confusion_matrix_with_precision_recall
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import math
import os


# _________________________________________________________________________________________________________________________
# _________________________________________________________________________________________________________________________
# _________________________________________________________________________________________________________________________


# General variables

path = '/home/s2412003/Shared/JAIST_Cylinder/Synchronized_Dataset'
do_train = True
do_plot = False

# Seed for reproducibility
np.random.seed(0)

# Model parameters
nhead = 16
num_encoder_layers = 2
dim_feedforward = 256

# Initialized later
input_dim = 0
output_dim = 0
max_seq_length = 0

# Training and Evaluation
num_epochs = 100
learning_rate = 0.00005
batch_size = 16
patience = 10


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

sensor_idxs = { 'Hand':0, 
                'Index_Intermediate':1, 'Index_Proximal':2, 
                'Middle_Intermediate':3, 'Middle_Proximal':4, 
                'Thumb_Distal':5, 'Thumb_Meta':6,
                'Wrist':7
                }

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

feature_conf = [x for x in range(9)]

checkpoint_model_name = f'IMU_checkpoint_model_{sensor_conf_name}_{learning_rate}lr_{batch_size}bs.pt'
confusion_matrix_name = f'IMU_confusion_matrix_{sensor_conf_name}_{learning_rate}lr_{batch_size}bs.png'

# _________________________________________________________________________________________________________________________
# _________________________________________________________________________________________________________________________
# _________________________________________________________________________________________________________________________



imu_files = []
imu_labels = []

trials = sorted(os.listdir(path))
for i in tqdm(range(len(trials))):
    trial_dir = trials[i]
    print(trial_dir)
    for hand_dir in os.listdir(os.path.join(path, trial_dir)):
        # print(f'\n')
        for action_dir in os.listdir(os.path.join(path, trial_dir, hand_dir)):
            print(f'\tTrial: {trial_dir}, Hand: {hand_dir}, Action: {action_dir}')
            temp_path = os.path.join(path, trial_dir, hand_dir, action_dir)
            # print(f'\t{temp_path}')

            # Load the IMU data
            imu_data = np.load(os.path.join(temp_path, f'{action_dir}.npz'))['imu']
            # print(f'\t{imu_data.shape=}')

            # Remove quaternion data
            imu_data = imu_data[:,:,4:]
            # print(f'\t{imu_data.shape=}')
            imu_data = imu_data[sensor_conf, :, :]
            imu_data = imu_data[:, :, feature_conf]
            # print(f'\t{imu_data.shape=}')
            # exit()
            # Go from (n_imus, sequence_length, n_features) to (sequence_length, n_imus, n_features)
            imu_data = imu_data.transpose(1, 0, 2)
            # print(f'\t{imu_data.shape=}')

            imu_data = imu_data.reshape(imu_data.shape[0], -1)
            # print(f'\t{imu_data.shape=}')
            imu_files.append(imu_data)
            imu_labels.append(action_dir)


print(f'\n{len(imu_files)} IMU files read.')
print(f'{len(imu_labels)} labels created.')


imu_sequencies, imu_labels, sequence_lengths = do_cut_actions(imu_files, imu_labels, action_cut_time_dict)

print(f'\n{len(imu_sequencies)} sequences created.'
      f'\n{len(imu_labels)} labels created.'
      f'\n{len(sequence_lengths)} sequence lengths created.') 



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

X_train, X_test, y_train, y_test, lengths_train, lengths_test = train_test_split(padded_sequences,
                                                                                 label_indices, 
                                                                                 sequence_lengths, 
                                                                                 test_size=0.3, 
                                                                                 random_state=0)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train)
lengths_train = torch.tensor(lengths_train)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test)
lengths_test = torch.tensor(lengths_test)

print(f'\n{X_train.shape=}, {y_train.shape=}, {lengths_train.shape=}, {X_train.dtype=}')
print(f'{X_test.shape=}, {y_test.shape=}, {lengths_test.shape=}, {X_test.dtype=}')

train_dataset = SequenceDataset(X_train, y_train, lengths_train)
test_dataset = SequenceDataset(X_test, y_test, lengths_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Reinitialize some parameters for the model
input_dim = padded_sequences.shape[2]
output_dim = len(action_names)
max_seq_length = padded_sequences.shape[1]

model = HAR_Transformer(input_dim, nhead, num_encoder_layers, dim_feedforward, output_dim, max_seq_length)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Initialize early stopping
early_stopping = EarlyStopper(saving_path=os.path.join('IMU_results', checkpoint_model_name), patience=patience)

best_model = None
train_losses = []

# Training loop
if do_train:
    for epoch in range(num_epochs):
        model.train()
        for i, (_, features, labels, lengths) in enumerate(train_loader):
            features, labels, lengths = features.to(device), labels.to(device), lengths.to(device)
            
            outputs = model(features, lengths)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Best Loss: {early_stopping.min_validation_loss:.4f}, Counter: {early_stopping.counter}')

        train_losses.append(loss.item())

        if early_stopping.early_stop(loss.item(), model.state_dict()):
            print("Early stopping")

            # Save the best model
            # torch.save(model.state_dict(), os.path.join('IMU_results', checkpoint_model_name))

            break
        else:
            best_model = model.state_dict()

    # Plot train losses
    plt.figure()
    plt.plot(train_losses)
    plt.title("Train Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()


print(f'Using device: {device}')

# Load the best model
model.load_state_dict(torch.load(os.path.join('IMU_results', checkpoint_model_name)))
model.to(device)
print(f'Model device: {next(model.parameters()).device}')


# Evaluation
model.eval()
y_true = []
y_pred = []

with torch.no_grad():
    for _, features, labels, lengths in test_loader:
        features, labels, lengths = features.to(device), labels.to(device), lengths.to(device)
        outputs = model(features, lengths)
        _, predicted = torch.max(outputs.data, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

accuracy = accuracy_score(y_true, y_pred)
print(f'Test Accuracy: {accuracy:.4f}')

# Confusion matrix with precision and recall
conf_matrix_ext = create_confusion_matrix_with_precision_recall(y_true, y_pred)

# Compute F1 score using scikit-learn
f1 = f1_score(y_true, y_pred, average='macro')
print(f'F1 Score: {f1:.4f}')


# # Confusion matrix
# conf_matrix = confusion_matrix(y_true, y_pred)

# # Calculate precision and recall for each class
# precision = precision_score(y_true, y_pred, average=None)
# recall = recall_score(y_true, y_pred, average=None)

# # From the test_labels, create a dictionary with the number of samples per class
# test_labels_dict = {}
# for label in y_true:
#     if label not in test_labels_dict:
#         test_labels_dict[label] = 0
#     test_labels_dict[label] += 1

# # print(f'\n{test_labels_dict=}')

# # Normalize the confusion matrix by the number of samples per class
# # print(f'\n{conf_matrix=}')
# for i in range(conf_matrix.shape[0]):
#     # print(f'{conf_matrix[i, :]}, {test_labels_dict[i]}\n {conf_matrix[i, :] / test_labels_dict[i]}')
#     conf_matrix[i, :] = conf_matrix[i, :] / test_labels_dict[i] * 100
# # print(f'\n{conf_matrix=}')

# # Append precision and recall to the confusion matrix
# recall = recall * 100
# precision = precision * 100
# accuracy = accuracy * 100

# conf_matrix_ext = np.c_[conf_matrix, recall]
# recall_ext = np.append(precision, accuracy)  # Add a nan for the last cell in recall row
# conf_matrix_ext = np.vstack([conf_matrix_ext, recall_ext])


# Plot extended confusion matrix
plt.figure(figsize=(16, 9))
sns.heatmap(conf_matrix_ext, annot=True, fmt='.2f', cmap='Blues', xticklabels= action_names + ['Recall'], yticklabels= action_names + ['Precision'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix with Precision and Recall')
plt.savefig(os.path.join('IMU_results', confusion_matrix_name.split('.')[0] + f'_f1:{f1}.png'))
plt.show()
