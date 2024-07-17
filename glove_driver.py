#!/usr/bin/env python3
from tkinter.font import names
import struct
import socket
import argparse
import os
import json, requests
import pickle
import keyboard
from IMU_class import IMU, Quaternion, Vector3
from queue import Queue
import time

'''LA CLASSE CHE GESTISCE LA COMUNICAZIONE SOCKET E LA RICEZIONE DEI DATI'''
class ImuDriver:

    def __init__(self, hand):
        self.in_udp_ip = '192.168.137.1' #'130.251.13.158' #STATIC IP ETHERNET CABLE; '192.168.0.101' #Hard-coded static IP
        self.in_udp_port = 2395


        self.msg = IMU('', Quaternion(0,0,0,0), Vector3(0,0,0), Vector3(0,0,0), Vector3(0,0,0))

        self.counter = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
 
        self.name_conversion_R = {
                                    'MC3': 'Wrist',
                                    'MC1': 'Thumb_Meta', 'PD1': 'Thumb_Distal',
                                    'PP2': 'Index_Proximal', 'PM2': 'Index_Intermediate', 
                                    'PP3': 'Middle_Proximal', 'PM3': 'Middle_Intermediate', 
                                    'PP4': 'Hand'
                                }

        self.name_conversion_L = {
                                    'MC3': 'Wrist',
                                    'PP5': 'Thumb_Meta', 'PM5': 'Thumb_Distal',
                                    'PP2': 'Middle_Proximal', 'PM2': 'Middle_Intermediate', 
                                    'PP3': 'Index_Proximal', 'PM3': 'Index_Intermediate', 
                                    'PP4': 'Hand'
                                }

        name_conversion = {}
        if hand == 'right':
            self.name_conversion = self.name_conversion_R
        else:
            self.name_conversion = self.name_conversion_L

        with open("config/driver_config.txt") as f:
            lines = f.readlines()
        
        self.names = {}
        for line in lines:
            values = line.strip().split(" ")
            self.names[int(values[0])] = values[1]


        self.namelist = sorted([n for n in self.names.values() if 'wrist' not in n])
        self.connectedIMUs = {n:0 for n in self.name_conversion.values()}

        self.msgCounter = 0

        self.in_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.in_sock.bind((self.in_udp_ip, self.in_udp_port))
        self.in_sock.settimeout(10)

        self.out_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        print(f"IMU listener initialized on IP {self.in_udp_ip} with ports {self.in_udp_port=}")


    def listener(self, thread_killer):
        try:
            data = self.in_sock.recv(29+6)  # (29) # buffer size is 1024 bytes
        except socket.timeout as se:
            print("Socket timeout", se)
            return None, False
        if not data:
            print("no data")
        else:
            ID = data[28+6]


            mux = int(ID/100)
            channel = int((ID-mux*100)/10)
            address = int(ID-mux*100-channel*10)


            self.msg.orientation.x = struct.unpack('<f', data[0:4])[0]
            self.msg.orientation.y = struct.unpack('<f', data[4:8])[0]
            self.msg.orientation.z = struct.unpack('<f', data[8:12])[0]
            self.msg.orientation.w = struct.unpack('<f', data[12:16])[0]

            self.msg.acceleration.x = struct.unpack('<h', data[16:18])[0]
            self.msg.acceleration.y = struct.unpack('<h', data[18:20])[0]
            self.msg.acceleration.z = struct.unpack('<h', data[20:22])[0]

            self.msg.angular_velocity.x = struct.unpack('<h', data[22:24])[0]
            self.msg.angular_velocity.y = struct.unpack('<h', data[24:26])[0]
            self.msg.angular_velocity.z = struct.unpack('<h', data[26:28])[0]

            self.msg.magnetic_field.x = struct.unpack('<h', data[28:30])[0]
            self.msg.magnetic_field.y = struct.unpack('<h', data[30:32])[0]
            self.msg.magnetic_field.z = struct.unpack('<h', data[32:34])[0]

            self.msg.sensor_id = self.name_conversion[self.names[ID]]# + "-Pose"


            tmp = (str(time.time()) + "," +
                str(self.msg.sensor_id) + "," + 
                str(self.msg.orientation.x) + "," + 
                str(self.msg.orientation.y) + "," + 
                str(self.msg.orientation.z) + "," + 
                str(self.msg.orientation.w) + "," +

                str(self.msg.acceleration.x) + "," +
                str(self.msg.acceleration.y) + "," +
                str(self.msg.acceleration.z) + "," +

                str(self.msg.angular_velocity.x) + "," +
                str(self.msg.angular_velocity.y) + "," +
                str(self.msg.angular_velocity.z) + "," +

                str(self.msg.magnetic_field.x) + "," + 
                str(self.msg.magnetic_field.y) + "," + 
                str(self.msg.magnetic_field.z) +  
                "\n")
            # print(tmp, end='\r')

            # sensor_queue.put(tmp)
            # self.connectedIMUs[self.name_conversion[self.names[ID]]] = 1
            # self.msgCounter += 1
            # # print(self.msgCounter, end='\r')
            # # print(self.name_conversion[self.names[ID]])
            # if self.msgCounter == 100:
            #     self.connectedIMUs = {n:0 for n in self.name_conversion.values()}
            #     self.msgCounter = 0
            # elif  self.msgCounter > 30:
            #     print(self.connectedIMUs)

            return tmp, True
            # return True
        
    def __del__(self):
        self.in_sock.close()
        print("IMU sockets closed.")


    def listener_loop(self, sensor_queue, thread_killer):
        timestamps = []
        previous_timestamp = 0.0
        while not thread_killer:
            timestamp = time.time()
            self.listener(thread_killer)
            if len(timestamps) == 0:
                timestamps.append(0.0)
                previous_timestamp = timestamp
            else:
                diff = timestamp - previous_timestamp
                previous_timestamp = timestamp
                timestamps.append(diff)
                # print(f'{diff}', end='\r')
            # try:
            #     print(f'{1/(sum(timestamps)/len(timestamps))}', end='\r')
            # except:
            #     pass
        print("Stopping listener loop...")
        print("Frequency:", frequency)


def main(): 
    imu_driver = ImuDriver('left')
    imu_driver.listener_loop(Queue(), False)


if __name__ == '__main__':
    main()

