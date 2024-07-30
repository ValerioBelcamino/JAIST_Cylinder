import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
from utils import RotateCircularPortion, CutBlackContour, tensor_to_pil  
import cv2
import numpy as np

# Define the custom transform class



# Define the transform pipeline
transform = transforms.Compose([
    # RotateCircularPortion(center=(323, 226), radius=210, random_angle= np.random.uniform(-180, 180)),  # Example center and radius
    CutBlackContour(left_margin=100, right_margin=65, top_margin=20, bottom_margin=0),
    # CutBlackContour(left_margin=80, right_margin=80, top_margin=0, bottom_margin=40),
    transforms.Resize((224, 224)),
    # transforms.ToTensor(),
    # transforms.Grayscale(num_output_channels=1),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # transforms.Normalize(mean=[0.0805], std=[0.1151]), #cam1
    # transforms.Normalize(mean=[0.0738], std=[0.1104]), #cam2
])

# Load an image
image_path = 'C:\\Users\\valer\\OneDrive\\Desktop\\JAIST_Cylinder\\Dataset\\0\\right\\push\\2\\1720407515.895586.jpg'
image_path2 = 'C:\\Users\\valer\\OneDrive\\Desktop\\JAIST_Cylinder\\Dataset\\0\\right\\push\\1\\1720407515.990453.jpg'
# image_path = 'C:\\Users\\valer\\OneDrive\\Desktop\\JAIST_Cylinder\\Dataset\\0\\left\\rub\\1\\1720405054.448848.jpg'
# image_path2 = 'C:\\Users\\valer\\OneDrive\\Desktop\\JAIST_Cylinder\\Dataset\\0\\right\\push\\1\\1720407515.990453.jpg'
image = Image.open(image_path)
image2 = Image.open(image_path)

# Apply the transform
transformed_image = transform(image)
transformed_image2 = transform(image2)

# print(f' max: {transformed_image.max()}')
# print(f' min: {transformed_image.min()}')
# print(transformed_image)
# print(transformed_image.shape)


# Visualize the result
# transformed_pil_image = tensor_to_pil(transformed_image)

x=112
y=112
r=110
print(f'x: {x}, y: {y}, r: {r}')
draw = ImageDraw.Draw(transformed_image)
leftUpPoint = (x-r, y-r)
rightDownPoint = (x+r, y+r)
twoPointList = [leftUpPoint, rightDownPoint]
draw.ellipse(twoPointList, outline=(255,0,0,1))

x=112
y=112
r=5
print(f'x: {x}, y: {y}, r: {r}')
draw = ImageDraw.Draw(transformed_image)
leftUpPoint = (x-r, y-r)
rightDownPoint = (x+r, y+r)
twoPointList = [leftUpPoint, rightDownPoint]
draw.ellipse(twoPointList, outline=(255,0,0,1))

plt.figure(figsize=(10, 5))

# Original Image
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image)

# Transformed Image
plt.subplot(1, 2, 2)
plt.title('Transformed Image')
plt.imshow(transformed_image,  cmap='gray')
# print(transformed_image.shape)
# print(transformed_image.permute(1,2,0).shape)
plt.show()