import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F


class sloth:
    def __init__(self, model, window_size, class_size, rho, tau, c, action_dict_inv):
        self.model = model
        self.window_size = window_size
        self.class_size = class_size
        self.probabilities_size = window_size

        self.action_dict_inv = action_dict_inv

        self.probabilities = np.empty((1,self.probabilities_size,self.class_size))
        self.probabilities[:] = np.nan

        self.rho = [rho for i in range(class_size)]
        self.tau = [tau for i in range(class_size)]
        self.c = [c for i in range(class_size)]

        self.time = 0
        self.peaks = np.zeros((1,self.class_size))

        self.gestures = []

    def classify(self, X):
        self.probabilities = np.roll(self.probabilities,self.probabilities_size-1,1)
        self.probabilities[0,-1,:] = F.softmax(self.model(*X), dim=1).cpu().detach().numpy()


    def detect(self):
        delta_prob = (self.probabilities[0,-1,:] - self.probabilities[0,-1-1,:]) 
        possible_peaks = np.where(delta_prob > self.rho)
        possible_peaks = possible_peaks[0]

        for ids in possible_peaks:
            if self.peaks[0, ids] == 0:
                self.peaks[0, ids] = self.time
            else:
                time_diff = self.time - self.peaks[0, ids]
                if time_diff >= self.c[ids]:
                    self.peaks[0, ids] = self.time
        active_peaks = np.where(self.peaks[0,:]> 0)
        active_peaks = active_peaks[0]

        for ids in active_peaks:
            time_diff = self.time - self.peaks[0, ids] + 1
            if time_diff >= self.c[ids]:
                start = int(self.probabilities_size-time_diff)
                prob_mean = np.mean(self.probabilities[0,start:,ids])
                if prob_mean > self.tau[ids]:
                    self.peaks[0, ids] = 0
                    self.gestures.append(ids+1)
                    print(f'Gesture {self.action_dict_inv[ids]} detected')
        self.time += 1

    def window_update(self, x, y, z):
        self.time += 1

    def display(self):
        plt.clf()
        plt.figure(1)
        plt.subplot(911)
        plt.plot(range(0,self.probabilities_size),self.probabilities[0,:,0])
        plt.axis([0, self.probabilities_size, -0.5, 1.5])
        plt.subplot(912)
        plt.plot(range(0,self.probabilities_size),self.probabilities[0,:,1])
        plt.axis([0, self.probabilities_size, -0.5, 1.5])
        plt.subplot(913)
        plt.plot(range(0,self.probabilities_size),self.probabilities[0,:,4])
        plt.axis([0, self.probabilities_size, -0.5, 1.5])
        plt.subplot(914)
        plt.plot(range(0,self.probabilities_size),self.probabilities[0,:,5])
        plt.axis([0, self.probabilities_size, -0.5, 1.5])
        plt.subplot(915)
        plt.plot(range(0,self.probabilities_size),self.probabilities[0,:,2])
        plt.axis([0, self.probabilities_size, -0.5, 1.5])
        plt.subplot(916)
        plt.plot(range(0,self.probabilities_size),self.probabilities[0,:,3])
        plt.axis([0, self.probabilities_size, -0.5, 1.5])

        plt.ion()
        plt.pause(0.05)

    def get_gesures(self):
        temp = self.gestures
        self.gestures = []
        return temp