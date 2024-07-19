import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
import torch
import cv2 


# Early stopping class
class EarlyStopper:
    def __init__(self, saving_path, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')
        self.best_model_state_dict = None
        self.saving_path = saving_path

    def early_stop(self, validation_loss, model_state_dict):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.best_model_state_dict = model_state_dict
            print(f'Saving model to {self.saving_path}')
            torch.save(self.best_model_state_dict, self.saving_path)
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


# Function to cut the actions
def do_cut_actions(data_list, label_list, action_cut_time_dict):
    new_data_list = []
    new_label_list = []
    new_lengths = []
    
    for data, label in zip(data_list, label_list):
        cut_time = action_cut_time_dict[label]
        sub_action_length = cut_time * 30
        n_sequences = data.shape[0] // sub_action_length
        
        last = 0
        for i in range(n_sequences):
            new_data_list.append(data[i*sub_action_length:(i+1)*sub_action_length])
            new_label_list.append(label)
            new_lengths.append(sub_action_length)
            last = (i+1)*sub_action_length
        if (data.shape[0] - last) > (sub_action_length / 2):
            new_data_list.append(data[last:])
            new_label_list.append(label)
            new_lengths.append(data.shape[0] - last)
    return new_data_list, new_label_list, new_lengths


def do_cut_actions_with_videos(imu_data_list, cam1_data_list, cam2_data_list, label_list, action_cut_time_dict):
    new_imu_data_list = []
    new_cam1_data_list = []
    new_cam2_data_list = []
    new_label_list = []
    new_lengths = []
    
    for imu_data, cam1_data, cam2_data, label in zip(imu_data_list, cam1_data_list, cam2_data_list, label_list):
        cut_time = action_cut_time_dict[label]
        sub_action_length = cut_time * 30
        n_sequences = imu_data.shape[0] // sub_action_length
        
        last = 0
        for i in range(n_sequences):
            new_imu_data_list.append(imu_data[i*sub_action_length:(i+1)*sub_action_length])
            new_cam1_data_list.append(cam1_data[i*sub_action_length:(i+1)*sub_action_length])
            new_cam2_data_list.append(cam2_data[i*sub_action_length:(i+1)*sub_action_length])
            new_label_list.append(label)
            new_lengths.append(sub_action_length)
            last = (i+1)*sub_action_length
        if (imu_data.shape[0] - last) > (sub_action_length / 2):
            new_imu_data_list.append(imu_data[last:])
            new_cam1_data_list.append(cam1_data[last:])
            new_cam2_data_list.append(cam2_data[last:])
            new_label_list.append(label)
            new_lengths.append(imu_data.shape[0] - last)
    return new_imu_data_list, new_cam1_data_list, new_cam2_data_list, new_label_list, new_lengths


# Function to pad stranded sequences
def do_pad_stranded_sequencies(stranded_seqs):
    max_length = max([len(seq) for seq in stranded_seqs])
    print(f'Maximum sequence length: {max_length/30} seconds (={max_length} samples).')

    padded_seqs = np.zeros((len(stranded_seqs), max_length, stranded_seqs[0].shape[1]))
    print(f'Allocated an array of shape {padded_seqs.shape}.')
    for i, seq in enumerate(stranded_seqs):
        # if seq.shape[1] != 72:
        #     print(i, seq.shape)
        #     continue
        # print(f'{i=}, {seq.shape=}')
        padded_seqs[i, :seq.shape[0], :] = seq
        # if seq.shape[0] < max_length:
        #     print(padded_seqs[i, seq.shape[0], :])
    return padded_seqs


class RotateCircularPortion:
    def __init__(self, center, radius, random_angle=0):
        self.center = center
        self.radius = radius
        self.random_angle = random_angle
    
    def __call__(self, img):
        # Convert PIL image to NumPy array
        # img = np.array(img)
        # print(f'img max: {img.max()}')
        # Get the circular portion
        circular_portion, mask = self.get_circular_portion(img[0])
        # Generate a random angle for rotation
        
        # print(f'random_angle: {self.random_angle}')
        
        # Rotate the circular portion
        rotated_circular_portion = self.rotate_image(circular_portion, self.random_angle)
        
        # Overlay the rotated circular portion back onto the original image
        img[:, :, :] = 0
        # print(f'img max: {img.max()}')
        # print(f'img shape: {img.shape}')
        # print(f'img shape: {rotated_circular_portion.shape}')
        # print(f'img shape: {mask.shape}')
        img[0][mask == 255] = rotated_circular_portion[mask == 255]
        
        # Convert NumPy array back to PIL image
        # img = Image.fromarray(img)

        # print(f'img shape: {img.shape}')
        # print(f'img max: {img.max()}')
        return torch.from_numpy(img)

    def get_circular_portion(self, image):
        # Create a mask with a filled circle
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.circle(mask, self.center, self.radius, (255), thickness=-1)
        
        # Extract the circular portion
        circular_portion = cv2.bitwise_and(image, image, mask=mask)
        
        return circular_portion, mask

    def rotate_image(self, image, angle):
        (h, w) = image.shape[:2]
        
        # Perform the rotation
        M = cv2.getRotationMatrix2D(self.center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))
        
        return rotated

class CutBlackContour:
    def __init__(self, left_margin, right_margin, top_margin, bottom_margin):
        self.left_margin = left_margin
        self.right_margin = right_margin
        self.top_margin = top_margin
        self.bottom_margin = bottom_margin
    
    def __call__(self, img):
        # Convert PIL image to NumPy array
        img = np.array(img)
        # print(f'image shape: {img.shape}')

        img = img[:, self.left_margin:-self.right_margin]
        # print(f'image shape: {img.shape}')
        return Image.fromarray(img)


# Convert the transformed tensor back to a PIL image for visualization
def tensor_to_pil(tensor):
    tensor = tensor.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    tensor = std * tensor + mean
    tensor = np.clip(tensor, 0, 1)
    return Image.fromarray((tensor * 255).astype(np.uint8))


def play_video(video_np, fps=30):
    # Convert the tensor to a NumPy array and remove the batch dimension
    video_array = video_np.permute(0, 2, 3, 1).numpy()  # Shape: [60, 224, 224, 3]

    # Normalize the array to [0, 255] and convert to uint8
    video_array = (video_array - video_array.min()) / (video_array.max() - video_array.min()) * 255
    video_array = video_array.astype(np.uint8)

    # Display the video at X fps
    delay = int(1000 / fps)  # Delay between frames in milliseconds

    for frame in video_array:
        cv2.imshow('Video', frame)
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


class SeededRandomSampler(Sampler):
    def __init__(self, data_source, seed):
        self.data_source = data_source
        self.seed = seed

    def __iter__(self):
        np.random.seed(self.seed)
        indices = np.random.permutation(len(self.data_source)).tolist()
        return iter(indices)

    def __len__(self):
        return len(self.data_source)