import cv2
import threading
import os
from datetime import datetime
import time
from queue import Queue
import signal
import inspect
from glove_driver import ImuDriver


def signal_handler(sig, frame):
    ''' Handle the SIGINT signal (Ctrl+C) by stopping the threads and exiting the program. '''
    global thread_killer, driver_threads
    if not thread_killer:
        print('Ctrl+C detected. Stopping threads...')
        thread_killer = True
        
        
    

def IMUs_driver(sensor_queues, hand):
    ''' Start the IMU driver to listen for IMU data. '''
    global trial, action, thread_killer, save_imu_thread_started
    imu_driver = ImuDriver(hand)

    while True:
        imu_data, not_socket_timeout = imu_driver.listener(thread_killer)
        
        if save_imu_thread_started:
            # print(imu_data)
            sensor_id = imu_data.split(',')[1]
            # print(sensor_id)
            sensor_queues[sensor_id].put(imu_data)

        if thread_killer or not not_socket_timeout:
            for sensor_name in imu_sensor_names:
                sensor_queues[sensor_name].put(None)
            break
    print(f'Stopping thread {inspect.stack()[0][3]}...')


def save_imu_data_by_sensor_name(save_dir, sensor_queue, sensor_name):
    ''' Save the IMU data to a file. '''
    print(f'Thread {inspect.stack()[0][3]} started for sensor {sensor_name}...')
    global trial, action, thread_killer, save_imu_thread_started
    dir_path = os.path.join(save_dir, 'imu_data')
    # print(f'Saving IMU data to {dir_path}...')

    assert os.path.exists(dir_path)

    save_imu_thread_started = True

    queue_trials = 0
    while True:
        
        imu_data = sensor_queue.get()
        # print(len(imu_data))
        if imu_data is None:  # Stop signal
            print('QUEUE EMPTY...')
            break

        queue_trials = 0

        # sensor_id = imu_data.split(',')[1]
        ts = imu_data.split(',')[0]
        imu_data = ','.join(imu_data.split(',')[2:])
        # if imu_data is None:  # Stop signal
        #     break
        timestamp = ts
        filename = os.path.join(dir_path, f"{sensor_name}.csv")

        imu_data = ','.join([timestamp, imu_data])

        with open(filename, 'a+') as f:
            f.write(imu_data)
        sensor_queue.task_done()

        # if thread_killer:
        #     break

    print(f'Stopping thread {inspect.stack()[0][3]}...{sensor_name}')

def check_sensors(fps=2):
    ''' Check if the sensors are on and print the status.'''
    global videoOn1, videoOn2, thread_killer
    interval = 1.0/fps
    while True: 
        start_time = time.time()
        print(f'videoOn1: {int(videoOn1)} videoOn2: {int(videoOn2)}')

        # Control the frame rate
        elapsed_time = time.time() - start_time
        if elapsed_time < interval:
            time.sleep(interval - elapsed_time)

        if thread_killer:
            break

    print(f'Stopping thread {inspect.stack()[0][3]}...')


def save_frames(video_id, frame_queue, save_dir):
    """Save frames from the queue to the specified directory."""
    global trial, action, save_image_thread_started_1, save_image_thread_started_2

    dir_path = os.path.join(save_dir, str(video_id))
    print(f'Saving frames to {dir_path}...')

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    if video_id == 1:
        save_image_thread_started_1 = True
    if video_id == 2:
        save_image_thread_started_2 = True

    while True:
        ret = frame_queue.get()
        if ret is None:  # Stop signal
            break
        ts, frame = ret[0], ret[1]
        timestamp = ts #str(time.time())
        filename = os.path.join(dir_path, f"{timestamp}.jpg")

        cv2.imwrite(filename, frame)
        frame_queue.task_done()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print(f'Stopping thread {inspect.stack()[0][3]} for video {video_id}...')


def capture_video(frame_queue1, frame_queue2, fps=30):
    """Capture video frames from the specified video source and add them to the queue."""
    global videoOn1, videoOn2, thread_killer, save_image_thread_started_1, save_image_thread_started_2
    vid1 = cv2.VideoCapture(2)
    vid2 = cv2.VideoCapture(1)
    interval = 1.0 / fps
    
    while vid1.isOpened() and vid2.isOpened():
        videoOn1, videoOn2 = True, True
        # print(f'vid1: {vid1.isOpened()} vid2: {vid2.isOpened()}')
        # start_time = time.time()
        if  vid1:
            ret1, frame1 = vid1.read()
        else:
            continue
        if  vid2:
            ret2, frame2 = vid2.read()
        else:
            continue
        if ret1 and save_image_thread_started_1:
            frame_queue1.put((time.time(), frame1))
        if ret2 and save_image_thread_started_2:
            frame_queue2.put((time.time(), frame2))
        cv2.imshow(f'frame_{1}', frame1)
        cv2.imshow(f'frame_{2}', frame2)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if thread_killer:
            break

        # Control the frame rate
        # elapsed_time = time.time() - start_time
        # if elapsed_time < interval:
        #     time.sleep(interval - elapsed_time)

    videoOn1, videoOn2 = False, False
    
    vid1.release()
    vid2.release()
    cv2.destroyAllWindows()
    frame_queue1.put(None)  # Send stop signal to saving thtread
    frame_queue2.put(None)  # Send stop signal to saving thread
    print(f'Stopping thread {inspect.stack()[0][3]}...')


def main():
    global driver_threads, trial, action, hand

    # Directories to save the data
    save_dir = f'C:\\Users\\valer\\OneDrive\\Desktop\\JAIST_Cylinder\\Dataset'

    sensor_folders = ['1', '2', 'imu_data']

    # Create the directories if they do not exist
    save_dir_trial_action = os.path.join(save_dir, trial, hand, action)
    if not os.path.exists(save_dir_trial_action):
        os.makedirs(save_dir_trial_action)
    for sensor_folder in sensor_folders:
        if not os.path.exists(os.path.join(save_dir_trial_action, sensor_folder)):
            os.makedirs(os.path.join(save_dir_trial_action, sensor_folder))

    # imu_driver = ImuDriver()
    # imu_driver.listener(trial, action)
    # exit()

    
    # Queues to hold frames captured from each camera
    frame_queue_0 = Queue()
    frame_queue_1 = Queue()
    imu_queues = {}
    
    for sensor_name in imu_sensor_names:
        imu_queues[sensor_name] = Queue()

    # Create and start threads for capturing video
    video_capture_thread = threading.Thread(target=capture_video, args=(frame_queue_0, frame_queue_1, 30))

    # Create and start threads for saving frames
    save_thread_0 = threading.Thread(target=save_frames, args=(1, frame_queue_0, save_dir_trial_action))
    save_thread_1 = threading.Thread(target=save_frames, args=(2, frame_queue_1, save_dir_trial_action))

    # Create and start thread for IMUs
    imu_thread = threading.Thread(target=IMUs_driver, args=(imu_queues, hand))

    save_imu_threads = []
    # Create and start thread for saving IMU data
    for sensor_name in imu_sensor_names:
        save_imu_threads.append(threading.Thread(target=save_imu_data_by_sensor_name, args=(save_dir_trial_action, imu_queues[sensor_name], sensor_name)))

    check_sensors_thread = threading.Thread(target=check_sensors, args=())

    driver_threads.extend([
                            video_capture_thread,
                            imu_thread
                         ])

    saver_threads = []
    saver_threads.extend(save_imu_threads)
    saver_threads.extend([save_thread_0, save_thread_1])

    # threads.extend([video_capture_thread,  
    #                 imu_thread,
    #                # check_sensors_thread,
    #                 save_thread_0, 
    #                 save_thread_1,
    #                 save_imu_thread])


    signal.signal(signal.SIGINT, signal_handler)

    # Start driver threads
    for thread in driver_threads:
        thread.start()
    
    # Wait input before recording
    input("press to continue\n")

    # Start saving threads
    for thread in saver_threads:
        thread.start()

    time.sleep(60)
    signal_handler(None, None)
    # while True:
    #     time.sleep(1)

    for thread in driver_threads:
            thread.join()
            print(f'Thread {thread.name} stopped.')
    for thread in saver_threads:
            thread.join()
            print(f'Thread {thread.name} stopped.')

    print('All threads stopped.')


if __name__ == "__main__":
    # Define global variables
    # imu_sensor_names = ['MC3-Pose', 'PM2-Pose', 'PP2-Pose', 'PM3-Pose', 'PP3-Pose', 'PM4-Pose', 'PP4-Pose', 'PM5-Pose', 'PP5-Pose']
    imu_sensor_names = ['Wrist', 'Thumb_Meta', 'Thumb_Distal', 'Index_Proximal', 'Index_Intermediate', 'Middle_Proximal', 'Middle_Intermediate', 'Hand']
    driver_threads = []
    videoOn1, videoOn2 = False, False
    save_image_thread_started_1 = False
    save_image_thread_started_2 = False
    save_imu_thread_started = False
    thread_killer = False
    trial = "6"
    action = "rub"
    hand = 'right'

    main()

   
