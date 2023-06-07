import numpy as np
import matplotlib
import pandas as pd
from electromyography import *


def calc_emg_features(emg):
    from EMG import EMGprep
    
    fs = 700 


    # all_emg_features = np.asarray(np.zeros(18), dtype = "float")

    all_emg_features = pd.DataFrame()
    
    window = int(0.5*60*fs)
    
    
    t_tot = (len(emg)//(int(window)))
    emg_tot = np.zeros([window, t_tot])
    emg_filt = np.zeros([window, t_tot])
    
    
    
    for i in range(t_tot):
        emg_data = emg[i*int(window):(i+1)*int(window)]
        t = np.arange(0, len(emg_data)*(1/fs), (1/fs))
        t = t[:len(emg_data)]
        emg_tot[:, i] = emg_data
    
    
    for i in range(t_tot):
        EMG = EMGprep(fs, emg_tot[:,i], "")
        emg_filtered = EMG.filtering_data()
        matplotlib.pyplot.close('all')
        emg_filt[:, i] = emg_filtered
        
    for i in range(t_tot):
        emg_features = analyzeEMG(emg_filt[:,i], fs, preprocessing=False, threshold=0.01)

        del emg_features['TimeDomain']['HIST']
        # del emg_features['TimeDomain']['MAV1']
        # del emg_features['TimeDomain']['MAV2']
        # del emg_features['TimeDomain']['TM3']
        # del emg_features['TimeDomain']['TM4']
        # del emg_features['TimeDomain']['TM5']
        # del emg_features['TimeDomain']['AFB']
        del emg_features['TimeDomain']['MAVSLPk']
        # del emg_features['FrequencyDomain']['TTP']
        # del emg_features['FrequencyDomain']['SM1']
        # del emg_features['FrequencyDomain']['SM2']
        # del emg_features['FrequencyDomain']['SM3']
        # del emg_features['FrequencyDomain']['FR']
        # del emg_features['FrequencyDomain']['VCF']

        emg_pd_time_features = pd.DataFrame([emg_features['TimeDomain']])
        emg_pd_freq_features = pd.DataFrame([emg_features['FrequencyDomain']])

        # features = []

        # for emg_id, emg_info in emg_features.items():    
        #     for key in emg_info:
        #         features = np.append(features, emg_info[key])
                                
        # all_emg_features = np.vstack((all_emg_features, features))
        emg_pd_features = pd.concat([emg_pd_freq_features, emg_pd_time_features], axis = 1)


        all_emg_features = pd.concat([all_emg_features, emg_pd_features], ignore_index=True)
        
        
    return all_emg_features
