import os 
import numpy as np
from PIL import Image
from utils import CutBlackContour
from torchvision import transforms
import torchvision.transforms as transforms

path = 'C:\\Users\\valer\\OneDrive\\Desktop\\JAIST_Cylinder\\Dataset'

images1 = []
images2 = []

transform = transforms.Compose([
    CutBlackContour(left_margin=80, right_margin=80, top_margin=0, bottom_margin=0),
    transforms.ToTensor(),
    transforms.Grayscale(num_output_channels=1),
])

for trial in os.listdir(path):
    for hand in os.listdir(os.path.join(path, trial)):
        for action in os.listdir(os.path.join(path, trial, hand)):
            print(f'{trial=}, {hand=}, {action=}')
            i = 0
            for cam1 in os.listdir(os.path.join(path, trial, hand, action, '1')):
                if i % 10 == 0:
                    img1 = os.path.join(path, trial, hand, action, '1', cam1)
                    img1 = Image.open(os.path.join(path, trial, hand, action, '1', img1))
                    img1 = transform(img1)
                    images1.append(img1)
                i += 1
                
            j = 0
            for cam2 in os.listdir(os.path.join(path, trial, hand, action, '2')):
                if j % 10 == 0:
                    img2 = os.path.join(path, trial, hand, action, '2', cam2)
                    img2 = Image.open(os.path.join(path, trial, hand, action, '2', img2))
                    img2 = transform(img2)
                    images2.append(img2)
                j += 1

print(f'len(images1): {len(images1)}')
print(f'len(images2): {len(images2)}')      

avg_image1 = np.stack(images1, axis=0)
avg_image2 = np.stack(images2, axis=0)

avg1 = np.mean(avg_image1, axis=(0, 2, 3))
avg2 = np.mean(avg_image2, axis=(0, 2, 3))
std1 = np.std(avg_image1, axis=(0, 2, 3))
std2 = np.std(avg_image2, axis=(0, 2, 3))

print(f'{avg1=}')
print(f'{avg2=}')
print(f'{std1=}')
print(f'{std2=}')


# Compute the average image
# avg_image1 = avg_image1 / len(img1)
# avg_image2 = avg_image2 / len(img2)