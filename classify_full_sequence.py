import numpy as np
from models import CHARberoViVit, HAR_Transformer, BicefHARlo
import torch
from torchvision import transforms
from utils import play_video
import time
import os 

def load_labels(path):
    labels = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            labels.append(int(line.split(',')[1]))
    return np.array(labels)

base_path = 'Z:\\Shared\\JAIST_Cylinder'
base_path = '/home/s2412003/Shared/JAIST_Cylinder'
path = f'{base_path}/Segmented_Dataset_Realtime'
folders = ['Labels', 'Video1', 'Video2', 'IMU']

action_names = ['linger', 'massaging', 'patting', 
                'pinching', 'press', 'pull', 
                'push', 'rub', 'scratching', 
                'shaking', 'squeeze', 'stroke', 
                'tapping', 'trembling']

action_dict = {action: i for i, action in enumerate(action_names)}
action_dict_inv = {i: action for i, action in enumerate(action_names)}

# Initialized later
nhead = 8
num_encoder_layers = 2
dim_feedforward = 128
intermediate_dim = 64
pixel_dim = 224
patch_size = 56
max_time = 150
n_features = 72

which_model = 'imu'




video1_list = []
video2_list = []
imu_list = []
labels_list = []

for folder in folders:
    if folder == 'Video1':
        video1_list = os.listdir(os.path.join(path, folder))
    elif folder == 'Video2':
        video2_list = os.listdir(os.path.join(path, folder))
    elif folder == 'IMU':
        imu_list = os.listdir(os.path.join(path, folder))
    elif folder == 'Labels':
        labels_list = os.listdir(os.path.join(path, folder))

print('\n')
print(f'Video1: {video1_list}\n')
print(f'Video2: {video2_list}\n')
print(f'IMU: {imu_list}\n')
print(f'Labels: {labels_list}\n')



# Use label file to match the video files with the IMU files

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

if which_model == 'both':
    model = CHARberoViVit(
                        pixel_dim, 
                        patch_size, 
                        len(action_names), 
                        max_time, 
                        n_features, 
                        nhead, 
                        num_encoder_layers, 
                        dim_feedforward, 
                        intermediate_dim
                        ).to(device)
    model.load_state_dict(torch.load(f'{base_path}/video_imu_results/checkpoint_model_IMUdoubleVideo_0.0001lr_16bs_224px_56ps_FalseAug.pt'))

elif which_model == 'imu':
    nhead = 16
    dim_feedforward = 256

    model = HAR_Transformer(n_features, nhead, num_encoder_layers, dim_feedforward, len(action_names), max_time).to(device)
    model.load_state_dict(torch.load(f'{base_path}/IMU_results/IMU_checkpoint_model_all_imus_0.0005lr_32bs.pt'))

elif which_model == 'videos':
    model = BicefHARlo(
                        pixel_dim, 
                        patch_size, 
                        len(action_names), 
                        max_time, 
                        intermediate_dim
                        ).to(device)
    model.load_state_dict(torch.load(f'{base_path}/video_results/checkpoint_model_VideoDoubleModel_0.0001lr_8bs_224px_56ps_FalseAug.pt'))

print('Model State Loaded\n') 


# model_IMU.eval()
# model = 

print('Model in evaluation mode\n')

# Create a tensor with the value 150 to provide as batch length to the model
batch_length = torch.tensor([150]).to(device)

transform = transforms.Compose([  
                transforms.Normalize(mean=[0.03728], std=[0.0750]),  
            ])

for lab in labels_list:
    print('\n')
    length = lab.split('_')[0]
    video1 = [v for v in video1_list if length in v][0]
    video2 = [v for v in video2_list if length in v][0]
    imu = [i for i in imu_list if length in i][0]

    video1_np = np.load(os.path.join(path, 'Video1', video1))
    video2_np = np.load(os.path.join(path, 'Video2', video2))

    print('Video Loaded\n')
    print(f'{video1_np.shape=}, {video2_np.shape=}\n')

    # play_video(video1_np)


    video1_list_temp = []
    video2_list_temp = []
    for i in range(video1_np.shape[0]):
        video1_list_temp.append(transform(torch.from_numpy(video1_np[i])))
    for i in range(video2_np.shape[0]):
        video2_list_temp.append(transform(torch.from_numpy(video2_np[i])))

    video1_tensor = torch.stack(video1_list_temp, dim=1)
    video2_tensor = torch.stack(video2_list_temp, dim=1)

    video1_tensor = video1_tensor.permute(1,0,2,3)
    video2_tensor = video2_tensor.permute(1,0,2,3)


    imu_tensor = torch.from_numpy(np.load(os.path.join(path, 'IMU', imu)))
    labels_tensor = torch.from_numpy(load_labels(os.path.join(path, 'Labels', lab)))

    print(f'{video1_tensor.shape=}, {video2_tensor.shape=}, {imu_tensor.shape=}, {labels_tensor.shape=}\n')

    # Create another tensor for classification labels of type int and same shape as labels_tensor
    output_labels = torch.zeros_like(labels_tensor)

    lenn = 90

    # define tensor windows
    video1_window = torch.zeros((lenn, 1, pixel_dim, pixel_dim)).unsqueeze(0)
    video2_window = torch.zeros((lenn, 1, pixel_dim, pixel_dim)).unsqueeze(0)
    imu_window = torch.zeros((lenn, n_features)).unsqueeze(0)
    print(f'{video1_window.shape=}, {video2_window.shape=}, {imu_window.shape=}\n')

    video1_window = video1_window.to(device)
    video2_window = video2_window.to(device)
    imu_window = imu_window.to(device)
    print('Moved windows to device\n')

    # Put the first elements in the windows
    video1_window[:, -1, :, :, :] = video1_tensor[0, :, :, :]
    video2_window[:, -1, :, :, :] = video2_tensor[0, :, :, :]
    imu_window[:, -1, :] = imu_tensor[0, :]
    print('First elements in windows\n')


    window_pad_video = torch.zeros((max_time-lenn, 1, pixel_dim, pixel_dim)).unsqueeze(0).to(device)
    window_pad_imu = torch.zeros((max_time-lenn, n_features)).unsqueeze(0).to(device)

    video1_window_new = torch.cat((video1_window, window_pad_video), dim=1)
    video2_window_new = torch.cat((video2_window, window_pad_video), dim=1)
    imu_window_new = torch.cat((imu_window, window_pad_imu), dim=1)

    if which_model == 'both':
        outputs = model(video1_window, video2_window, imu_window, batch_length)
    elif which_model == 'imu':
        # print(f'shape: {imu_window_new.shape}')
        outputs = model(imu_window_new, batch_length)
    elif which_model == 'videos':
        outputs = model(video1_window_new, video2_window_new)

    _, predicted = torch.max(outputs.data, 1)
    print(f'\n{action_dict_inv[predicted.item()]}\n')
    output_labels[0] = predicted


    # Iterate over the rest of the elements
    for i in range(1, len(labels_tensor)):
        # Roll the windows
        video1_window = torch.roll(video1_window, shifts=-1, dims=1)
        video2_window = torch.roll(video2_window, shifts=-1, dims=1)
        imu_window = torch.roll(imu_window, shifts=-1, dims=1)

        # Put the new elements in the windows
        video1_window[:, -1, :, :, :] = video1_tensor[i, :, :, :]
        video2_window[:, -1, :, :, :] = video2_tensor[i, :, :, :]
        imu_window[:, -1, :] = imu_tensor[i, :]

        window_pad_video = torch.zeros((max_time-lenn, 1, pixel_dim, pixel_dim)).unsqueeze(0).to(device)
        window_pad_imu = torch.zeros((max_time-lenn, n_features)).unsqueeze(0).to(device)

        video1_window_new = torch.cat((video1_window, window_pad_video), dim=1)
        video2_window_new = torch.cat((video2_window, window_pad_video), dim=1)
        imu_window_new = torch.cat((imu_window, window_pad_imu), dim=1)

        if which_model == 'both':
            outputs = model(video1_window, video2_window, imu_window, batch_length)
        elif which_model == 'imu':
            # print(f'shape: {imu_window_new.shape}')
            outputs = model(imu_window_new, batch_length)
        elif which_model == 'videos':
            outputs = model(video1_window_new, video2_window_new)

        _, predicted = torch.max(outputs.data, 1)
        print(f'\n{i/30.0:.2f}: {action_dict_inv[predicted.item()]}\n')
        output_labels[i] = predicted



    # Save output_labels to a file
    if not os.path.exists(os.path.join(base_path, 'realtime_results')):
        os.makedirs(os.path.join(base_path, 'realtime_results'))

    with open(os.path.join(base_path, 'realtime_results', f'{length}_predicted_labels.txt'), 'w') as f:
        for item in output_labels:
            f.write(f'{item.item()}\n')
