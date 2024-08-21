import torch
from torch.utils.data import Dataset, DataLoader
from utils import RotateCircularPortion, CutBlackContour, tensor_to_pil
from PIL import Image
from torchvision import transforms
import numpy as np


def pad_videos(video, max_length):
    if video.shape[1] <= max_length:
        zeros = torch.zeros(video.shape[0], max_length, video.shape[2], video.shape[3])
        zeros[:, :video.shape[1], :, :] = video
        return zeros

class VideoDataset(Dataset):
    def __init__(self, image_names, labels, lengths, max_length=300, pixel_dim=224, cam_id=1):
        self.image_names = image_names
        # print(f'len(image_names): {len(image_names)}')
        self.labels = labels
        # print(f'len(labels): {len(labels)}\n')
        self.lengths = lengths
        self.max_length = max_length
        self.pixel_dim = pixel_dim
        self.cam_id = cam_id

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        # print(f'{self.image_names[idx]=}')
        sequence = self.load_sequence(self.image_names[idx])
        
        label = self.labels[idx]
        
        # return idx
        return idx, sequence, label

    def load_sequence(self, action_image_names):
        # Load all PNG files in the folder and sort them
        images = action_image_names

        sequence = []
        for image in images:
            try:
                sequence.append(Image.open(image))
            except:
                print(f'Error opening {image}')
        # sequence = [Image.open(img) for img in images]
        
        # Toss a coin to decide whether to flip the video horizontally
        horizontal_flip = np.random.rand() < 0.2

        # Apply any necessary transforms (e.g., resize, normalization)

        if self.cam_id == 1:
            transform = transforms.Compose([
                # RotateCircularPortion(center=(323, 226), radius=210, random_angle= np.random.uniform(-180, 180)),  # Example center and radius
                CutBlackContour(left_margin=100, right_margin=65, top_margin=20, bottom_margin=0),
                transforms.Resize((self.pixel_dim, self.pixel_dim)),
                transforms.Grayscale(num_output_channels=1),    
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.0738], std=[0.1104]),  
                # transforms.Lambda(lambda img: transforms.functional.hflip(img) if horizontal_flip else img),
            ])
        elif self.cam_id == 2:
            transform = transforms.Compose([
                # RotateCircularPortion(center=(323, 226), radius=210, random_angle= np.random.uniform(-180, 180)),  # Example center and radius
                CutBlackContour(left_margin=80, right_margin=80, top_margin=0, bottom_margin=40),
                transforms.Resize((self.pixel_dim, self.pixel_dim)),
                transforms.Grayscale(num_output_channels=1),    
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.0738], std=[0.1104]),  
                # transforms.Lambda(lambda img: transforms.functional.hflip(img) if horizontal_flip else img),
            ])
        
        sequence = [transform(img) for img in sequence]
        # sequence = [transforms.functional.pil_to_tensor(transform(img)) for img in sequence]
        # print(sequence[0].dtype)
        sequence = torch.stack(sequence, dim=1)
        # print(f'{sequence.shape=}')
        sequence = pad_videos(sequence, self.max_length)
        # print(f'{sequence.shape=}')
        sequence = sequence.permute(1,0,2,3)
        # print(f'{sequence.shape=}')
        return sequence  # Shape: (T, C, H, W)




class VideoDatasetNPY(Dataset):
    def __init__(self, image_names, labels, lengths, video_augmentation, max_length=300, pixel_dim=224, cam_id=1):
        self.image_names = image_names
        # print(f'len(image_names): {len(image_names)}')
        self.labels = labels
        # print(f'len(labels): {len(labels)}\n')
        self.lengths = lengths
        self.max_length = max_length
        self.pixel_dim = pixel_dim
        self.video_augmentation = video_augmentation
        self.cam_id = cam_id

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        # print(f'{self.image_names[idx]=}')
        sequence = self.load_sequence(self.image_names[idx])
        label = self.labels[idx]
        # return idx
        return sequence, label

    def load_sequence(self, action_image_names):
        sequence = np.load(action_image_names)
        # print(f'{sequence.shape=}')
        # Toss a coin to decide whether to flip the video horizontally
        horizontal_flip = np.random.rand() < 0.2

        random_angle = np.random.uniform(-180, 180)

        new_sequence = []
        if self.video_augmentation:
            # print('Augmenting video...')
            # Apply any necessary transforms (e.g., resize, normalization)
            if self.cam_id == 1:
                transform = transforms.Compose([
                    # transforms.Normalize(mean=[0.07427], std=[0.09232]),  
                    # transforms.Normalize(mean=[-0.2251], std=[0.0939]),  
                    # transforms.Normalize(mean=[0.2959], std=[0.9831]),  
                    RotateCircularPortion(center=(112, 112), radius=110, random_angle=random_angle, channel_first=True),
                ])
            if self.cam_id == 2:
                transform = transforms.Compose([
                    # transforms.Normalize(mean=[0.081102], std=[0.09952]),
                    # transforms.Normalize(mean=[-0.218], std=[0.1012]),  
                    # transforms.Normalize(mean=[0.3505], std=[1.061]),  
                    RotateCircularPortion(center=(112, 112), radius=110, random_angle=-random_angle, channel_first=True),
                ])

            for i in range(sequence.shape[0]):
                new_sequence.append(transform(torch.from_numpy(sequence[i])))
            # new_sequence = [torch.from_numpy(sequence[i]) for i in range(sequence.shape[0])]
            # sequence = [torch.from_numpy(sequence[i]) for i in range(sequence.shape[0])]
        else:
            # print('Not augmenting video...')

            # cam_id == 2        
            # 
            if self.cam_id == 1:
                transform = transforms.Compose([  
                    transforms.Normalize(mean=[0.07427], std=[0.09232]),  
                    # transforms.Normalize(mean=[-0.2251], std=[0.0939]),  
                    # transforms.Normalize(mean=[0.2959], std=[0.9831]),  
                ])

            elif self.cam_id == 2:
                transform = transforms.Compose([  
                    transforms.Normalize(mean=[0.081102], std=[0.09952]),
                    # transforms.Normalize(mean=[-0.218], std=[0.1012]),
                    # transforms.Normalize(mean=[0.3505], std=[1.061]),
                ])
            # print(f'{sequence[0].shape=}')
            for i in range(sequence.shape[0]):
                # new_sequence.append(transform(torch.from_numpy(sequence[i])))
                new_sequence.append(torch.from_numpy(sequence[i]))
            # print(f'{new_sequence[0].shape=}')
        sequence = torch.stack(new_sequence, dim=1)
        # print(f'{sequence.shape=}')
        # exit()
        # Print max and min
        # print(f'{sequence.max()}')
        # print(f'{sequence.min()}')
        sequence = pad_videos(sequence, self.max_length)
        sequence = sequence.permute(1,0,2,3)
        return sequence  


