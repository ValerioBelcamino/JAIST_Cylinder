import os
import numpy as np
from PIL import Image
path = '/home/s2412003/Shared/JAIST_Cylinder/Synchronized_Dataset'
# path = 'Z:\\Shared\\JAIST_Cylinder\\Synchronized_Dataset'

#set numpy print options infitinite precision
np.set_printoptions(precision=10)

# x = 'Z:\\Shared\\JAIST_Cylinder\\Synchronized_Dataset\\0\\left\\linger\\images_1'
# x_imf = '1720404637.7329085.jpg'
# print(os.listdir(x))
# print(len(os.listdir(x)))
# y = Image.open(os.path.join(x, x_imf))
errors = 0
for trial in sorted(os.listdir(path)):
    if trial != '0':
        break
    for hand in ['left', 'right']:
        for action in sorted(os.listdir(f'{path}/{trial}/{hand}')):
            current_folder = os.path.join(path, trial, hand, action)
            np_data = np.load(os.path.join(current_folder, f'{action}.npz'))
            image_names1 = np_data['camera1']
            image_names2 = np_data['camera2']

            # #check if same
            # for i, (im1, im2) in enumerate(zip(image_names1, image_names2)):
            #     if im1 != im2:
            #         print(f'{im1} != {im2}')
            #         errors += 1

            # print(f'{image_names1[:10]}')
            # print(f'{image_names2[:10]}')
            # exit()
            for i in range(len(image_names1)):
                # print(i)
                img_path = os.path.join(current_folder, 'images_1')
                images_in_folder = [k[:-4] for k in os.listdir(img_path)]
                try:
                    Image.open(os.path.join(img_path, str(image_names1[i])+'.jpg'))
                except:
                    print(f'aaaError opening {os.path.join(img_path, str(image_names1[i])+".jpg")}')
                    errors += 1
            #     # if str(image_names2[i]) not in images_in_folder:
            #     #     print(f'{image_names1[i]} not in {img_path}')

            for j in range(len(image_names2)):
                # print(j)
                img_path = os.path.join(current_folder, 'images_2')
                images_in_folder = [k[:-4] for k in os.listdir(img_path)]
                try:
                    Image.open(os.path.join(img_path, str(image_names2[j])+'.jpg'))
                except:
                    print(f'Error opening {os.path.join(img_path, str(image_names1[j])+".jpg")}')
                    errors += 1

                # if str(image_names2[j]) not in images_in_folder:
                #     print(f'{image_names2[j]} not in {img_path}')
print(f'Errors: {errors}')