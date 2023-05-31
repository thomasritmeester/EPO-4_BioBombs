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
from IPython.display import display

##################################
#Creating a function:
def ECG_freq_data(ecg):
    from ECG import ECGprep

    ecg_features = np.asarray(np.zeros(4), dtype = "float")


    fs = 700 #sampling freq.

    #########################################################
    #cut a smaller window
    wdw=int(0.5*60*fs)
    size_adpt=int(len(ecg)/(len(ecg)/(int(wdw))))

    #print(size_adpt, "size of the samples")
    t_tot = (len(ecg)//(int(wdw)))
    ecg_tot = np.zeros([size_adpt, t_tot])
    #ecg_base_tot = np.zeros([size_adpt, t_tot])
    #print(t_tot, 'ttot1')

    for i in range(t_tot):
        ecg1 = ecg[i*int(wdw):(i+1)*int(wdw)]
        #ecg2 = base[i*int(wdw):(i+1)*int(wdw)]
        t1 = np.arange(0, ecg1.size*(1/fs), (1/fs))
        t1 = t1[:ecg1.size]
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

    #for i in range(t_tot):
       # ECG_base = ECGprep(fs, ecg_base_tot[:,i], "baseline")
        # ECG = ECGprep(fs, ecg_tot[:,i], "stress")

        # #ecg_filt_b = ECG_base.filtering_data()
        # ecg_filt = ECG.filtering_data()

    #print(ecg_filt_b.shape, 'filtering shape')

    #######################################################
    #Feature extraction, obtaining the peaks and getting the HRV time domain data.
    #ECG_feat_base = pd.DataFrame()
    ECG_feat = pd.DataFrame()
    for i in range (t_tot):              #t_tot
        #ECG_feat_base = nk.ecg_clean( ecg_filt_b, sampling_rate=fs)
        ECG_feat= nk.ecg_clean( ecg_tot[:,i], sampling_rate=fs)

        #peaks_b, info_b = nk.ecg_peaks(ECG_feat_base, sampling_rate=fs, correct_artifacts=True)
        peaks, info = nk.ecg_peaks(ECG_feat, sampling_rate=fs, correct_artifacts=True)

        #HRV time domain features only, no frequency or nonlinear measurements.
        #hrv_b=nk.hrv_time(peaks_b, sampling_rate=fs, show=False)
        hrv=nk.hrv_frequency(peaks, sampling_rate=fs, show=False)

        hrv=hrv.dropna(1)
        hrv_numpy = hrv.to_numpy()
        #print("hrv:")
        # print(hrv)
        # print("numpy:")
        # print(hrv_numpy)
        # print(hrv_numpy.shape)

        ecg_features = np.vstack((ecg_features, hrv.to_numpy()[0,:]))
        #hrv_b=hrv_b.dropna()
        # print(hrv.shape, 'HRV 1')
        # display(hrv.to_string())
        
    return ecg_features[1:,:]
    # eda_features = np.vstack((ecg_features,ecg_feat_s))
    # eda_features = np.vstack((ecg_features,ecg_feat_b))
    #print(hrv_b, 'HRV 2')
    

# print('\n')
# print(ecg1.shape, ecg_stress_tot.shape, '1')
# print('\n')
# print(np.shape(ecg_filt_b), np.shape(ecg_filt_s),'2')
# #print(np.shape(ecg_feat_b), np.shape(ecg_feat_s))
 

 

