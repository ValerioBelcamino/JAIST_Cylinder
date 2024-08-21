import numpy as np
import os 
from utils import play_video, RotateCircularPortion
from torchvision import transforms
import torch

path = 'z:\\Shared\\JAIST_Cylinder\\Segmented_Dataset2'
path_rt = 'z:\\Shared\\JAIST_Cylinder\\Segmented_Dataset_Realtime'

video_a = os.path.join(path, '0', 'Video1', '0_13_90_video1.npy')
video_b = os.path.join(path, '0', 'Video2', '0_13_90_video2.npy')
video_rt_a = os.path.join(path_rt, 'Video1', '0_10780_video1.npy')
video_rt_b = os.path.join(path_rt, 'Video2', '0_10780_video2.npy')

channel_first = True
random_angle = np.random.randint(-180, 180)
transform1 = transforms.Compose([  
                    RotateCircularPortion(center=(112, 112), radius=110, random_angle=random_angle, channel_first=channel_first),
                    transforms.Normalize(mean=[0.07427], std=[0.09232]),  
                    # transforms.Normalize(mean=[-0.2251], std=[0.0939]),  
                    # transforms.Normalize(mean=[0.2959], std=[0.9831]),  
                ])
transform2 = transforms.Compose([  
                    RotateCircularPortion(center=(112, 112), radius=110, random_angle=random_angle, channel_first=channel_first),
                    transforms.Normalize(mean=[0.081102], std=[0.09952]),
                    # transforms.Normalize(mean=[-0.218], std=[0.1012]),
                    # transforms.Normalize(mean=[0.3505], std=[1.061]),
                ])

video_a = np.load(video_a)
video_b = np.load(video_b)
# video_rt_a = np.load(video_rt_a)
# video_rt_b = np.load(video_rt_b)

print(f'{video_a.shape=}, {video_b.shape=}')
# print(f'{video_rt_a.shape=}, {video_rt_b.shape=}')

# Permute  from (time, channel, height, width) to (time, height, width, channel)
# video_a = video_a.transpose(0, 2, 3, 1)
# video_b = video_b.transpose(0, 2, 3, 1)
video_a = video_a.transpose(0, 1, 2, 3)
video_b = video_b.transpose(0, 1, 2, 3)
# video_rt_a = video_rt_a.transpose(0, 2, 3, 1)
# video_rt_b = video_rt_b.transpose(0, 2, 3, 1)
print(f'{video_a.shape=}, {video_b.shape=}')
# print(f'{video_rt_a.shape=}, {video_rt_b.shape=}')

video_as = []
video_bs = []
for i in range(video_a.shape[0]):
    video_as.append(transform1(torch.from_numpy(video_a[i])))
    video_bs.append(transform2(torch.from_numpy(video_b[i])))
    # video_a[i] = transform1(torch.from_numpy(video_a[i]))
    # video_b[i] = transform2(torch.from_numpy(video_b[i]))

video_a = np.stack(video_as, axis=0)
video_b = np.stack(video_bs, axis=0)

# for i in range(video_rt_a.shape[0]):
#     video_rt_a[i] = transform1(torch.from_numpy(video_rt_a[i]))
#     video_rt_b[i] = transform2(torch.from_numpy(video_rt_b[i]))


play_video(video_a)
play_video(video_b)

# play_video(video_rt_a)
# play_video(video_rt_b)