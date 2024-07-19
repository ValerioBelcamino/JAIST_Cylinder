import os 
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import shutil

plt.ion()
do_plot = False
verbose = False
# hand = 'left'

def plot_timestamps(timestamps):
    plt.figure()
    for i in range(len(timestamps)):
        plt.plot(timestamps[i]) 
    
    plt.xlabel('Index')
    plt.ylabel('Timestamp')
    plt.title('Synchronized Timestamps')
    plt.legend(range(len(timestamps)))

    plt.show()


def load_image_names(image_folder_path):
    ''' Load the image names from the dataset path. '''
    return sorted([
                    float('.'.join(img.split('.')[:2])) 
                    for img in os.listdir(image_folder_path) 
                    if '.jpg' in img 
                    and 
                    os.path.isfile(os.path.join(image_folder_path, img))
                ])


def load_imu_data(imu_folder_path):
    ''' Load the IMU timestamps from the dataset path. '''
    imu_files = [f for f in os.listdir(imu_folder_path)
    if '.csv' in f
    and
    os.path.isfile(os.path.join(imu_folder_path, f))]
    timestamps_list = []
    data_cols_list = []
    for imu_f in sorted(imu_files):
        with open(os.path.join(imu_folder_path, imu_f), 'r') as file:
            timestamps = []
            data_cols = []
            lines = file.readlines()
            for line in lines:
                timestamps.append(float(line.split(',')[0]))
                data_cols.append(line.split(',')[1:])
            timestamps_list.append(sorted(timestamps))
            data_cols_list.append(data_cols)
    return timestamps_list, data_cols_list

base_path = '/home/s2412003/Shared/JAIST_Cylinder/'
dataset_path = os.path.join(base_path, 'Dataset')
# dataset_path = os.path.join(dataset_path, hand)
output_path = os.path.join(base_path, 'Synchronized_Dataset')

action_names = ['linger', 'massaging', 'patting', 
                'pinching', 'press', 'pull', 
                'push', 'rub', 'scratching', 
                'shaking', 'squeeze', 'stroke', 
                'tapping', 'trembling']

sensor_names = ['1', '2', 'imu_data']

specific_trial = 6

dir_content = os.listdir(dataset_path)

for i in tqdm(range(len(dir_content))):
    dir = dir_content[i]
    if int(dir) != specific_trial:
        print(f'Skipping trial {dir}...')
        print(specific_trial)
        continue
    trial_content = os.listdir(os.path.join(dataset_path, dir))
    for hand_name in ['left', 'right']:
        trial_content = os.listdir(os.path.join(dataset_path, dir, hand_name))
        print(trial_content)

        # assert set(trial_content) == set(action_names)

        for action_name in action_names:
            print(f'{dir} --> {hand_name} --> {action_name}')
            sensor_dirs = os.listdir(os.path.join(dataset_path, dir, hand_name, action_name))

            camera1_dir = os.path.join(dataset_path, dir, hand_name, action_name, '1')
            camera2_dir = os.path.join(dataset_path, dir, hand_name, action_name, '2')
            imu_data_dir = os.path.join(dataset_path, dir, hand_name, action_name, 'imu_data')
                
            camera1_ts = load_image_names(camera1_dir)
            camera2_ts = load_image_names(camera2_dir)
            imu_ts, imu_data = load_imu_data(imu_data_dir)
            

            # Create a list with all timestamps
            all_ts = []
            all_ts.append(camera1_ts)
            all_ts.append(camera2_ts)
            for ts in imu_ts:
                all_ts.append(ts)

            # Check if the timestamps are sorted
            for ts_list in all_ts:
                assert all(a <= b for a, b in zip(ts_list, ts_list[1:]))

            firsts = [ts[0] for ts in all_ts]
            lasts = [ts[-1] for ts in all_ts]

            # Find common start and end timestamps
            max_first = max(firsts)
            min_last = min(lasts)

            # print(f'Max first: {max_first}')
            # print(f'Min last: {min_last}')

            new_all_ts = []
            for ts_list in all_ts:
                new_all_ts.append([ts for ts in ts_list if ts >= max_first and ts <= min_last])

            if verbose:
                print('Timestamps before and after cutting:')
                for i, (ts_list, new_ts_list) in enumerate(zip(all_ts, new_all_ts)):
                    print(f'\t{i} - Old: {len(ts_list)} --> New: {len(new_ts_list)}')

            # Cut imu data to match the timestamps
            new_imu_data = []
            for i in range(len(imu_data)):
                new_imu_data.append([data for ts, data in zip(imu_ts[i], imu_data[i]) if ts >= max_first and ts <= min_last])

            if verbose:
                print('IMU data before and after cutting:')
                for i, (data_list, new_data_list) in enumerate(zip(imu_data, new_imu_data)):
                    print(f'\t{i} - Old: {len(data_list)} --> New: {len(new_data_list)}')

            # Check that the new lenght of the timestamps is the same as the new length of the imu data
            assert len(new_all_ts[2:]) == len(new_imu_data)
            for ts_list, data_list in zip(new_all_ts[2:], new_imu_data):
                assert len(ts_list) == len(data_list)

            # Convert the timestamps to numpy arrays
            for i in range(len(new_all_ts)):
                new_all_ts[i] = np.asarray(new_all_ts[i])# - max_first
            
            if do_plot:
                # Plot the synchronized timestamps
                plot_timestamps(new_all_ts)
            
            # Now synchronize the timestamps
            synchronized_all_ts = []
            synchronized_all_data = []
            synchronized_all_idxs = []
            for i in range(len(new_all_ts)):
                # Skip camera1, we are gonna use it as a reference to find closest timestamps
                if i == 0:
                    continue
                else:
                    # Find the closest timestamps
                    synchronized_ts = []
                    synchronized_idxs = []
                    for ts in new_all_ts[0]:
                    #     synchronized_ts.append(new_all_ts[i][np.argmin(np.abs(new_all_ts[i] - ts))])
                    # synchronized_all_ts.append(np.asarray(synchronized_ts))
                        synchronized_idxs.append(np.argmin(np.abs(new_all_ts[i] - ts)))
                    synchronized_all_idxs.append(np.asarray(synchronized_idxs))

            # Convert the synchronized indexes to timestamps and imu data
            for i in range(len(synchronized_all_idxs)):
                    synchronized_all_ts.append(np.asarray([new_all_ts[i+1][idx] for idx in synchronized_all_idxs[i]]))
            synchronized_all_ts.insert(0, new_all_ts[0])

            for i in range(len(new_imu_data)):
                    synchronized_all_data.append(np.asarray([new_imu_data[i][idx] for idx in synchronized_all_idxs[i+1]]))
                # synchronized_all_idxs[i] = synchronized_all_idxs[i] + 1

            if verbose:
                print('Timestamps before and after cutting:')
                for i, (ts_list, new_ts_list, synch_ts_list) in enumerate(zip(all_ts, new_all_ts, synchronized_all_ts)):
                    print(f'\t{i} - Old: {len(ts_list)} --> Cut: {len(new_ts_list)} --> Sync: {len(synch_ts_list)}')

                print('IMU data before and after cutting:')
                for i, (data_list, new_data_list, synch_imu_data_list) in enumerate(zip(imu_data, new_imu_data, synchronized_all_data)):
                    print(f'\t{i} - Old: {len(data_list)} --> Cut: {len(new_data_list)} --> Sync: {len(synch_imu_data_list)}')

            # Check that the new lenght of the timestamps is the same as the new length of the imu data
            assert len(synchronized_all_ts[2:]) == len(synchronized_all_data)
            for i in range(len(synchronized_all_ts)):
                assert len(synchronized_all_ts[0]) == len(synchronized_all_ts[i])
            for i in range(len(synchronized_all_data)):
                assert len(synchronized_all_ts[0]) == len(synchronized_all_data[i])

            if do_plot:
                # Plot the synchronized timestamps
                plot_timestamps(synchronized_all_ts)
                input('Press to continue...')

            n_samples = len(synchronized_all_ts[0])
            framerate = n_samples/(min_last - max_first)

            # Double check that the synchronized timestamps for the cameras correspond to existing images
            image_names1 = os.listdir(os.path.join(dataset_path, dir, hand_name, action_name, '1'))
            image_names2 = os.listdir(os.path.join(dataset_path, dir, hand_name, action_name, '2'))

            for i in range(n_samples):
                assert f'{synchronized_all_ts[0][i]}.jpg' in image_names1
                assert f'{synchronized_all_ts[1][i]}.jpg' in image_names2

            print(f'\t-> Average frame rate of the synchronized recordings: {framerate} fps')
            print(f'\t-> Average length of the synchronized recordings: {n_samples/framerate} seconds')
            
            # Create an npz file with the synchronized data
            npz_file = os.path.join(output_path, dir, hand_name, action_name, f'{action_name}.npz')
            os.makedirs(os.path.dirname(npz_file), exist_ok=True)
            # np.savez(npz_file, camera1=synchronized_all_ts[0], camera2=synchronized_all_ts[1], imu=synchronized_all_data)
            np.savez(npz_file, camera1=synchronized_all_ts[0], camera2=synchronized_all_ts[1], imu=synchronized_all_data, imu_ts=synchronized_all_ts[2:])
            
            # Now let's copy only the synchronized images to a new folder
            new_image_path_1 = os.path.join(output_path, dir, hand_name, action_name, 'images_1')
            new_image_path_2 = os.path.join(output_path, dir, hand_name, action_name, 'images_2')
            os.makedirs(new_image_path_1, exist_ok=True)
            os.makedirs(new_image_path_2, exist_ok=True)

            for i in tqdm(range(n_samples)):
                img1 = os.path.join(dataset_path, dir, hand_name, action_name, '1', f'{synchronized_all_ts[0][i]}.jpg')
                img2 = os.path.join(dataset_path, dir, hand_name, action_name, '2', f'{synchronized_all_ts[1][i]}.jpg')
                new_img1 = os.path.join(new_image_path_1, f'{synchronized_all_ts[0][i]}.jpg')
                new_img2 = os.path.join(new_image_path_2, f'{synchronized_all_ts[1][i]}.jpg')

                shutil.copy(img1, new_img1)
                shutil.copy(img2, new_img2)

            assert len(os.listdir(new_image_path_1)) == n_samples
            assert len(os.listdir(new_image_path_2)) == n_samples

            
            print(f'\t-> Synchronized data saved to {npz_file}...')


