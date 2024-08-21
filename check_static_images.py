import numpy as np
import os 
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as ssim

base_path = 'Z:\\Shared\\JAIST_Cylinder'
base_path = '/home/s2412003/Shared/JAIST_Cylinder'
path = f'{base_path}/Segmented_Dataset_Realtime'
folders = ['Labels', 'Video1', 'Video2', 'IMU']

def check_static_images(video, threshold=0.01):
    # Step 2: Initialize a list to store similarity scores
    similarity_scores = []

    for i in range(1, video.shape[0]):  # Start from the second frame (index 1)
        previous_frame = video[i - 1, 0, :, :]
        current_frame = video[i, 0, :, :]
        
        # Compute Mean Squared Error (MSE)
        mse = np.sum((current_frame - previous_frame))

        # if mse < threshold:
        #     similarity_scores.append(0)
        # else:
        #     similarity_scores.append(1)
        
        # Append the MSE score
        similarity_scores.append(mse)
    
    return similarity_scores



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

# Load the first video files
video1_np = np.load(os.path.join(path, 'Video1', video1_list[0]))
video2_np = np.load(os.path.join(path, 'Video2', video2_list[0]))
print(f'Video 1 shape: {video1_np.shape}')
print(f'Video 2 shape: {video2_np.shape}')


# Compute similarity scores for the first video
similarity_scores_video1 = check_static_images(video1_np)
similarity_scores_video2 = check_static_images(video2_np)


# Plot the similarity scores for the first video
plt.figure(figsize=(12, 6))
plt.plot(similarity_scores_video1, label='Video 1')
plt.plot(similarity_scores_video2, label='Video 2')
plt.xlabel('Frame Index')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('Similarity Scores for Video Frames')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join('similarity', 'similarity_scores.png'))
plt.show()

