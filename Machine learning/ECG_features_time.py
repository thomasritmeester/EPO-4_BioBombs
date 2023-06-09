#ECG

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
import pandas as pd
import neurokit2 as nk
#from IPython.display import display

##################################
#Creating a function:
def ECG_time_data(ecg):
    from ECG import ECGprep

    #ecg_features = np.asarray(np.zeros(18), dtype = "float")

    ecg_features = pd.DataFrame()

    fs = 700 #sampling freq.

    #########################################################
    #cut a smaller window
    wdw=fs*60*0.5
    size_adpt=(int(wdw))

    #print(size_adpt, "size of the samples")
    t_tot = int((len(ecg)//size_adpt))      #int(wdw)
    print("t_tot=" ,t_tot)
    ecg_tot = np.zeros([size_adpt, t_tot])   
    print("t_tot=",t_tot)
    #ecg_base_tot = np.zeros([size_adpt, t_tot])
    #print(t_tot, 'ttot1')

    for i in range(t_tot):
        ecg1 = ecg[i*int(wdw):(i+1)*int(wdw)]
        #ecg2 = base[i*int(wdw):(i+1)*int(wdw)]
        t1 = np.arange(0, len(ecg1)*(1/fs), (1/fs))
        t1 = t1[:len(ecg1)]
        # t2 = np.arange(0, ecg2.size*(1/fs), (1/fs))
        # t2 = t2[:ecg2.size]
        ecg_tot[:, i] = ecg1
        #ecg_base_tot[:, i] = ecg2

    # print(ecg2.shape, 'total base shape')
    # print(ecg_stress_tot.shape, 'Total stress')

    ####################################################
    #Data perperation, i.e. filtering etc
    #ECG_base = []
    ECG = []
    #print(t_tot, 'ttot2')


#############################################################
#Filtering the ECG data
    
    for i in range(t_tot):
        ECG = ECGprep(fs, ecg_tot[:,i], "stress")
        #print(ECG)
        ecg_filt = ECG.filtering_data()
    print("hallo")
    print("ecg_filt=", ecg_filt.shape)
    #######################################################
    #Feature extraction, obtaining the peaks and getting the HRV time domain data.
    for i in range (t_tot):   

        #ECG_feat_base = nk.ecg_clean( ecg_filt_b, sampling_rate=fs)
        #ECG_feat= nk.ecg_clean( ecg_filt, sampling_rate=fs)

        peaks, info = nk.ecg_peaks(ecg_filt, sampling_rate=fs, correct_artifacts=True)

        #HRV time domain features only, no frequency or nonlinear measurements.
        hrv=nk.hrv_time(peaks, sampling_rate=fs, show=False)

        #Dropping Nans
        hrv=hrv.dropna(axis = 'columns')
        #hrv = hrv.iloc[:,0:18]

        ecg_features = pd.concat([ecg_features, hrv], ignore_index=True)
        # hrv_numpy = hrv.to_numpy()[0,:18]

        # ecg_features = np.vstack((ecg_features, hrv_numpy))
        
    return ecg_features

 

 

