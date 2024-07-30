import cv2
import threading
import os
from datetime import datetime
import time
from queue import Queue
import numpy as np
import signal
import inspect
from glove_driver import ImuDriver
from torchvision import transforms
from utils import CutBlackContour
from models import CHARberoViVit, BicefHARlo, HAR_Transformer
import torch


def signal_handler(sig, frame):
    ''' Handle the SIGINT signal (Ctrl+C) by stopping the threads and exiting the program. '''
    global thread_killer
    if not thread_killer:
        print('Ctrl+C detected. Stopping threads...')
        thread_killer = True
        
        
    

def IMUs_driver(hand):
    ''' Start the IMU driver to listen for IMU data. '''
    global thread_killer, imu_queues
    imu_driver = ImuDriver(hand)

    while True:
        imu_data, not_socket_timeout = imu_driver.listener(thread_killer)
        
        sensor_id = imu_data.split(',')[1]
        imu_data_float = [float(x) for x in imu_data.split(',')[2:]]

        imu_queues[sensor_id] = imu_data_float

        if thread_killer or not not_socket_timeout:
            break
    print(f'Stopping thread {inspect.stack()[0][3]}...')



def IMUs_moving_window(fps = 30):
    ''' Reads the IMU data from the queues and creates a moving window of the data. '''
    global thread_killer, imu_queues, imu_sensor_names, imu_window_list, max_action_length, imu_feature_dim

    while True:
        start_time = time.time()

        features = torch.empty(imu_feature_dim)
        for i, sn in enumerate(sorted(imu_sensor_names)):
            features[i*9:(i+1)*9] = torch.tensor(imu_queues[sn])

        if len(imu_window_list) < max_action_length:
            imu_window_list.append(features)
        else:
            imu_window_list.pop(0)
            imu_window_list.append(features)

        if thread_killer:
            break

        # Control the frame rate
        elapsed_time = time.time() - start_time
        if elapsed_time < 1.0/fps:
            time.sleep(1.0/fps - elapsed_time)
        print(f'Elapsed time: {1/(time.time() - start_time):.4f}\n')


    print(f'Stopping thread {inspect.stack()[0][3]}...')


 
 

def capture_video_by_id(id, tf, fps=30):
    """Capture video frames from the specified video source and add them to the queue."""
    global videoOn, thread_killer, squared_resolution, frame_windows_lists, max_action_length
    vid = cv2.VideoCapture(id+1)
    interval = 1.0 / fps


    
    while vid.isOpened():
        videoOn[id] = True
        # print(f'vid1: {vid1.isOpened()} vid2: {vid2.isOpened()}')
        start_time = time.time()
        if vid:
            ret, frame = vid.read()
            frame_tf = tf(frame)
        else:
            continue
        
        if ret:
            if len(frame_windows_lists[id]) < max_action_length:
                frame_windows_lists[id].append(frame_tf)
            else:
                frame_windows_lists[id].pop(0)
                frame_windows_lists[id].append(frame_tf)
            # Roll the window buffer and fill last position with new frame
            # frame_windows[id] = torch.roll(frame_windows[id], shifts=-1, dims=1)
            # frame_windows[id][0, -1] = frame_tf

        cv2.imshow(f'frame_{id}', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if thread_killer:
            break

        # Control the frame rate
        # elapsed_time = time.time() - start_time
        # if elapsed_time < interval:
        #     time.sleep(interval - elapsed_time)
        # print(f'Elapsed time: {1/(time.time() - start_time)}')

    videoOn[id] = False
    
    vid.release()
    cv2.destroyWindow(f'frame_{id}')
    print(f'Stopping thread {inspect.stack()[0][3]}...')



def HAR_realtime_classification(imu_queues):
    global thread_killer, frame_windows_lists, max_action_length, videoOn, imu_feature_dim, imu_window_list, squared_resolution, action_names, which_model
    timer_fill_video_window = None
    timer_fill_imu_window = None

    done_filling_video_window = False
    done_filling_imu_window = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device}\n')

    if which_model == 'all':
        model = CHARberoViVit(
                            pixel_dim = squared_resolution, 
                            patch_size = 56, 
                            n_classes = len(action_names), 
                            max_seq_len = 150, 
                            n_features = imu_feature_dim, 
                            nhead = 8, 
                            num_encoder_layers = 2, 
                            dim_feedforward = 128, 
                            intermediate_dim = 64
                            )
    elif which_model == 'video':
        model = BicefHARlo(
                            pixel_dim = squared_resolution, 
                            patch_size = 56, 
                            n_classes = len(action_names), 
                            max_seq_len = 150, 
                            intermediate_dim = 64
                            )
    elif which_model == 'imu':
        model = HAR_Transformer(
                            input_dim = imu_feature_dim, 
                            nhead = 8, 
                            num_encoder_layers = 2, 
                            dim_feedforward = 128, 
                            output_dim = len(action_names), 
                            max_seq_length = 150
                            )

    model.to(device)
    model.eval()

    while True:
        loop_timer = time.time()
        if all(videoOn) and timer_fill_video_window is None:
            timer_fill_video_window = time.time()

        if timer_fill_imu_window is None:
            timer_fill_imu_window = time.time()
        
        print(f'frame_windows_lists: {len(frame_windows_lists)}')

        if (len(frame_windows_lists[0]) == max_action_length) and (len(frame_windows_lists[1]) == max_action_length):
            if not done_filling_video_window:
                print(f'time elapsed VIDEO: {time.time() - timer_fill_video_window:.4f}')
                done_filling_video_window = True
                # break

            video_window_np_0 = torch.unsqueeze(torch.stack(frame_windows_lists[0], axis=0), axis=0)
            video_window_np_1 = torch.unsqueeze(torch.stack(frame_windows_lists[1], axis=0), axis=0)

            video_window_np_0 = video_window_np_0.to(device)
            video_window_np_1 = video_window_np_1.to(device)

            print(f'window_np: {video_window_np_0.shape}')
            print(f'window_np: {video_window_np_1.shape}')

        if len(imu_window_list) == max_action_length:
            if not done_filling_imu_window:
                print(f'time elapsed IMU: {time.time() - timer_fill_imu_window:.4f}')
                done_filling_imu_window = True
                # break

            imu_window_np = torch.unsqueeze(torch.stack(imu_window_list, axis=0), axis=0)

            imu_window_np = imu_window_np.to(device)

            print(f'imu_window_np: {imu_window_np.shape}')

        # At you should have both Video and IMU windows filled as tensors


        print(f'Elapsed time: {(time.time() - loop_timer):.4f}\n')

        if thread_killer:
            break
    
    print(f'Stopping thread {inspect.stack()[0][3]}...')



def main():
    global driver_threads, max_action_length, squared_resolution, imu_feature_dim, imu_sensor_names

    transform = transforms.Compose([
        CutBlackContour(left_margin=80, right_margin=80, top_margin=0, bottom_margin=0),
        transforms.Resize((squared_resolution, squared_resolution)),
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=1),    
        transforms.Normalize(mean=[0.03728], std=[0.0750]),  
    ])

    # Create and start threads for capturing video
    video_capture_thread_1 = threading.Thread(target=capture_video_by_id, args=(0, transform, 30,))
    video_capture_thread_2 = threading.Thread(target=capture_video_by_id, args=(1, transform, 30,))
 
    # Create and start thread for IMUs
    imu_thread = threading.Thread(target=IMUs_driver, args=(imu_queues, hand))
    imu_moving_window_thread = threading.Thread(target=IMUs_moving_window, args=(30,))


    driver_threads.extend([
                            video_capture_thread_1,
                            video_capture_thread_2,
                            # imu_thread
                            # imu_moving_window_thread
                         ])
 
    signal.signal(signal.SIGINT, signal_handler)

    # Wait input before recording
    # input("press to continue\n")

    # Start driver threads
    for thread in driver_threads:
        thread.start()

    working_thread = threading.Thread(target=HAR_realtime_classification, args=(imu_queues,))
    working_thread.start()
    

    # Wait input before recording
    input("press to stop\n")

    # time.sleep(3)
    signal_handler(None, None)
    # while True:
    #     time.sleep(1)


    for thread in driver_threads:
            thread.join()
            print(f'Thread {thread.name} stopped.')

    working_thread.join()
    print('All threads stopped.')


if __name__ == "__main__":
    # Define global variables
    imu_sensor_names = ['Wrist', 'Thumb_Meta', 'Thumb_Distal', 'Index_Proximal', 'Index_Intermediate', 'Middle_Proximal', 'Middle_Intermediate', 'Hand']

    action_names = ['linger', 'massaging', 'patting', 
                    'pinching', 'press', 'pull', 
                    'push', 'rub', 'scratching', 
                    'shaking', 'squeeze', 'stroke', 
                    'tapping', 'trembling']

    driver_threads = []
    videoOn1, videoOn2 = False, False
    videoOn = [videoOn1, videoOn2]
    thread_killer = False
    max_action_length = 5 * 30
    squared_resolution = 224    
    n_imus = 8
    imu_feature_dim = 9 * n_imus
    hand = 'right'

    which_model = 'all'

    print(f'Starting with max_action_length: {max_action_length}, squared_resolution: {squared_resolution}, imu_feature_dim: {imu_feature_dim}, hand:{hand}', end='\n\n')

    # Create a window to hold the frames for each camera
    # shape: (batch_size, sequence_length, channels, height, width)
    # frame_window_1 = torch.empty((1, max_action_length, 1, squared_resolution, squared_resolution))
    # frame_window_2 = torch.empty((1, max_action_length, 1, squared_resolution, squared_resolution))

    # frame_window_1.fill_(float('nan'))
    # frame_window_2.fill_(float('nan'))

    # frame_windows = [frame_window_1, frame_window_2]
    frame_windows_lists = [[], []]

    imu_queues = {}
    for sensor_name in imu_sensor_names:
        imu_queues[sensor_name] = []

    imu_window_list = []

    # print(f'frame_window_1 initalized with {np.sum(np.isnan(frame_window_1))} NaNs.')
    # print(f'frame_window_2 initalized with {np.sum(np.isnan(frame_window_2))} NaNs.')

    main()

   
