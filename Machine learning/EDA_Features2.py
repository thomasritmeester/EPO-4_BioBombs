import os
import pickle
import numpy as np
from scipy import signal
from scipy.signal import butter, iirnotch, lfilter
import numpy as np
import matplotlib.pyplot as plt

from os import listdir
from os.path import isfile, join, isdir
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import scipy.ndimage
from EDA import EDAprep

#Extracting the data from the 
data_set_path = "C:/Users/semve/OneDrive/Documenten/WESAD/WESAD/"
subject = ["S2",'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S13', 'S14', 'S15', 'S16', 'S17']

class read_data_of_one_subject:
    """Read data from WESAD dataset"""
    def __init__(self, path, subject):
        self.keys = ['label', 'subject', 'signal']
        self.signal_keys = ['wrist', 'chest']
        self.chest_sensor_keys = ['ACC', 'ECG', 'EDA', 'EMG', 'Resp', 'Temp']
        self.wrist_sensor_keys = ['ACC', 'BVP', 'EDA', 'TEMP']
        #os.chdir(path)
        #os.chdir(subject)
        with open(path + subject +'/'+subject + '.pkl', 'rb') as file:
            data = pickle.load(file, encoding='latin1')
        self.data = data

    def get_labels(self):
        return self.data[self.keys[0]]

    def get_wrist_data(self):
        """"""
        #label = self.data[self.keys[0]]
        assert subject == self.data[self.keys[1]]
        signal = self.data[self.keys[2]]
        wrist_data = signal[self.signal_keys[0]]
        #wrist_ACC = wrist_data[self.wrist_sensor_keys[0]]
        #wrist_ECG = wrist_data[self.wrist_sensor_keys[1]]
        return wrist_data

    def get_chest_data(self):
        """"""
        signal = self.data[self.keys[2]]
        chest_data = signal[self.signal_keys[1]]
        return chest_data
    

#print(len(subject))
fs = 700

import numpy as np
from scipy.signal import chirp, find_peaks, peak_widths, peak_prominences

def calc_phasic_data(phasic, peak, height):
    ##Find all the points on the plot below the 50% of the peak
    half_points = np.where(((phasic - (phasic[peak] - 0.5*height) < 0.00001)))[0]

    ##Finds the index of the point directly to the right of the peak
    half_amp_index = np.inf
    for j in half_points:
        if(j < half_amp_index and j > peak):
            half_amp_index = j



    ##Calculate onset and offset
    onset_points = np.where(((phasic - (phasic[peak] - 0.63*height) < 0.00001)))[0]

    ##Finds the index of the point directly to the right of the peak
    offset_amp_index = np.inf
    for j in onset_points:
        if(j < offset_amp_index and j > peak):
            offset_amp_index = j

    ##Finds the index of the point directly to the right of the peak
    onset_amp_index = 0
    for j in onset_points:
        if(j > onset_amp_index and j < peak):
            onset_amp_index = j

  

    return onset_amp_index, half_amp_index, offset_amp_index


def calc_phasic_features(phasic, state):

    temp_array = np.asarray([], dtype = "float")


    orienting_mag = np.asarray([], dtype = "float")
    orienting_time = np.asarray([], dtype = "float")
    half_recov_time = np.asarray([], dtype = "float")

    peaks, _ = find_peaks(phasic)
    heights, _, __ = peak_prominences(phasic, peaks)
    widths, _, __, ___ = peak_widths(phasic, peaks, rel_height=0.63)

    # find the indices with an amplitude larger that 0.1
    keep = np.full(len(peaks), True)
    keep[peaks < 0.1] = False

    # only keep those
    peaks=peaks[keep]
    heights=heights[keep]
    widths=widths[keep]

    # peaks=np.hstack(peaks)
    # heights=np.hstack(heights)
    # widths=np.hstack(widths)

    for i in range(len(peaks)):
        onset_amp_index, half_amp_index, offset_amp_index = calc_phasic_data(phasic, peaks[i], heights[i])

        orienting_mag = np.append(orienting_mag, phasic[peaks[i]] - phasic[onset_amp_index])
        orienting_time = np.append(orienting_time, (offset_amp_index - onset_amp_index)/fs)
        half_recov_time = np.append(half_recov_time, (half_amp_index - peaks[i])/fs)

    cv_mg = np.std(orienting_mag)/np.mean(orienting_mag)
    cv_orient = np.std(orienting_time)/np.mean(orienting_time)
    cv_recov = np.std(half_recov_time)/np.mean(half_recov_time)

    temp_array = np.append(temp_array, np.mean(orienting_mag))   
    temp_array = np.append(temp_array, np.std(orienting_mag))
    temp_array = np.append(temp_array, cv_mg)
    temp_array = np.append(temp_array, np.mean(orienting_time))   
    temp_array = np.append(temp_array, np.std(orienting_time))
    temp_array = np.append(temp_array, cv_orient)
    temp_array = np.append(temp_array, np.mean(half_recov_time))   
    temp_array = np.append(temp_array, np.std(half_recov_time))
    temp_array = np.append(temp_array, cv_recov)


    return temp_array

def calc_eda_features(eda_data):

    eda_features = np.asarray([0,0,0,0,0,0,0,0,0], dtype = "float")

    fs=700
    # cut a smaller window
    t_tot=(len(eda_data)//(int(0.5*60*fs)))
    
    eda_data_tot=np.empty([21000,t_tot])

    for k in range(t_tot): 
        eda1=eda_data[k*int(0.5*60*700):(k+1)*int(0.5*60*700)]
        t1=np.arange(0,eda1.size*(1/fs),(1/fs))
        t1=t1[:eda1.size]
        eda_data_tot[:,k] = eda1
            
    #print(eda1.shape)
    #print(eda_stress_tot.shape)

    #print(eda2.shape)
    #print(eda_stress_tot.shape)

    eda_comp=np.zeros((3,11000,t_tot))
    EDA = []
    for j in range (t_tot): 
        EDA = EDAprep(fs, eda_data_tot[:,j],t_tot,"baseline")

        #EDA.plotdata()
        eda_lp = EDA.filtering_data()
        eda_sm = EDA.smoothing_data(eda_lp)
        eda_comp[:,:,j]=EDA.decompose_data(eda_sm)
        
        phasic = eda_comp[1][:,j]

        # plt.figure(figsize=(12,4))
        # plt.xlim([0,30])
        # plt.plot(t,phasic_stress,label='phasic')
        # plt.xlabel('$Time (s)$') 
        # plt.ylabel('$EDA$') 
        # plt.legend()

        #print((phasic_stress[0:10]))

        #For the state, True is stress and False is base
            
            
        feature = calc_phasic_features(phasic, True)
        #print(np.shape(stress_feature))

            
            
        eda_features = np.vstack((eda_features,feature))

    #print(eda_features_base)
    return eda_features[1:,:]
