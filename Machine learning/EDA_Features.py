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
import pandas as pd


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


def calc_phasic_features(phasic, eda_complete):

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

    #Calculate area under normalised phasic curve
    phasic_norm = (phasic - np.mean(phasic))/np.std(phasic)
    area_under = np.sum(np.abs(phasic_norm))
    minimum = np.amin(phasic)
    maximum = np.amax(phasic)
    drange = (maximum-minimum)


    temp_array = np.append(temp_array, np.mean(eda_complete))
    temp_array = np.append(temp_array, np.std(eda_complete))
    temp_array = np.append(temp_array, np.mean(phasic))
    temp_array = np.append(temp_array, np.std(phasic))
    temp_array = np.append(temp_array, len(peaks))
    temp_array = np.append(temp_array, area_under)
    temp_array = np.append(temp_array, drange)
    temp_array = np.append(temp_array, np.mean(orienting_mag))   
    temp_array = np.append(temp_array, np.std(orienting_mag))
    temp_array = np.append(temp_array, np.mean(orienting_time))   
    temp_array = np.append(temp_array, np.std(orienting_time))
    temp_array = np.append(temp_array, np.mean(half_recov_time))   
    temp_array = np.append(temp_array, np.std(half_recov_time))


    return temp_array

def calc_eda_features(eda_data):

    #eda_features = np.zeros(13)

    eda_features = pd.DataFrame()

    fs=700
    # cut a smaller window
    wndw = int(0.5*60*fs)
    t_tot=(len(eda_data)//(int(wndw)))

    
    eda_data_tot=np.empty([21000,t_tot])

    for k in range(t_tot): 
        eda1=eda_data[k*int(wndw):(k+1)*int((wndw))]
        t1=np.arange(0,len(eda1)*(1/fs),(1/fs))
        t1=t1[:len(eda1)]
        eda_data_tot[:,k] = eda1

        
            
    #print(eda1.shape)
    #print(eda_stress_tot.shape)

    #print(eda2.shape)
    #print(eda_stress_tot.shape)

    eda_comp=np.zeros((3,len(eda_data_tot[:,0]),t_tot))
    EDA = []
    for j in range (t_tot): 
        EDA = EDAprep(fs, eda_data_tot[:,j],"baseline")

        #EDA.plotdata()
        eda_lp = EDA.filtering_data()
        eda_sm = EDA.smoothing_data(eda_lp)
        eda_comp[:,:,j]=EDA.decompose_data(eda_sm)
        
        phasic = eda_comp[1][:,j]
        eda_complete = eda_comp[0][:,j]


        # plt.figure(figsize=(12,4))
        # plt.xlim([0,30])
        # plt.plot(t,phasic_stress,label='phasic')
        # plt.xlabel('$Time (s)$') 
        # plt.ylabel('$EDA$') 
        # plt.legend()

        #print((phasic_stress[0:10]))

        #For the state, True is stress and False is base
            
            
        feature = calc_phasic_features(phasic, eda_complete)
        #print(np.shape(stress_feature))

        feature_set=pd.DataFrame([feature], columns=['EDA_mean', 'EDA_std', 'Phasic_mean', 'Phasic_std', 'No.Peaks', 'Area', 'Dynamic_Range', 'Orienting_mag_mean', 'Orienting_mag_std', 'orient_time_mean', 'orient_time_std', 'recov_time_mean', 'recov_time_std'])

        eda_features = pd.concat([eda_features, feature_set], ignore_index=True)
            
            
        #eda_features = np.vstack((eda_features,feature))

   
    return eda_features
