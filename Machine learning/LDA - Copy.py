import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
from scipy.signal import chirp, find_peaks, peak_widths, peak_prominences
from scipy.signal import chirp, find_peaks, peak_widths, peak_prominences
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from EDA_Features import *
from Temperature_Features import *
from ECG_features_time import * 
from ECG_features_freq import *
from EMG_Features import *
from ACC_features import *
from wesad import read_data_of_one_subject
import warnings

from sklearn.utils import shuffle

warnings.simplefilter(action='ignore', category=FutureWarning)

#######################################################################
#Initializing the file, then making random train and test sets.
print("Start!")

data_set_path = "D:/Downloads/WESAD/WESAD/"
sensor_data = ["S2",'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S13', 'S14', 'S15', 'S16', 'S17']  
sub_shuf = shuffle(sensor_data)
print(sub_shuf)
train=sub_shuf[:14]
test=[sub_shuf[-1]]
print("Test subject=", test)

#######################################################################
#Reading out subjects and calling the feature extraction functions

def extraction (train_test):
    features_base = np.asarray(np.zeros(54), dtype = "float")
    features_stress = np.asarray(np.zeros(54), dtype = "float")
    
    for i in range(len(train_test)):     
        print("Subject: ", train_test[i])

        obj_data = {}

        obj_data[train_test[i]] = read_data_of_one_subject(data_set_path, train_test[i])
        #print(obj_data[subject[i]].data)
        chest_data_dict = obj_data[train_test[i]].get_chest_data()

        labels = obj_data[train_test[i]].get_labels() 
        baseline = np.asarray([idx for idx,val in enumerate(labels) if val == 1])
        stress = np.asarray([idx for idx,val in enumerate(labels) if val == 2])

        acc_chest_stress=chest_data_dict['ACC'][stress]

        eda_data_stress=chest_data_dict['EDA'][stress,0]
        eda_data_base=chest_data_dict['EDA'][baseline,0]

        emg_data_stress=chest_data_dict['EMG'][stress,0]
        emg_data_base=chest_data_dict['EMG'][baseline,0]
        
        temp_data_stress=chest_data_dict['Temp'][stress,0]
        temp_data_base=chest_data_dict['Temp'][baseline,0]

        ecg_data_stress=chest_data_dict['ECG'][stress,0]
        ecg_data_base=chest_data_dict['ECG'][baseline,0]    

        baseline_signals = [eda_data_base, emg_data_base, ecg_data_base]
        stress_signals = [eda_data_stress, emg_data_stress, ecg_data_stress]
        eda_data_base, emg_data_base, ecg_data_base, eda_data_stress, emg_data_stress, ecg_data_stress, acc_wrist_stress, acc_wrist_baseline = remove_movement(chest_data_dict, i, stress, baseline, baseline_signals, stress_signals)
        
        eda_features_base = calc_eda_features(eda_data_base)
        eda_features_stress = calc_eda_features(eda_data_stress)
        
        emg_features_base = calc_emg_features(emg_data_base)
        emg_features_stress = calc_emg_features(emg_data_stress)

        temp_features_base = calc_temp_features(temp_data_base)
        temp_features_stress = calc_temp_features(temp_data_stress)

        ecg_features_time_base = ECG_time_data(ecg_data_base)
        ecg_features_time_stress = ECG_time_data(ecg_data_stress)

        ecg_features_freq_base = ECG_freq_data(ecg_data_base)
        ecg_features_freq_stress = ECG_freq_data(ecg_data_stress)

        acc_features_stress = acc_features(acc_wrist_stress)
        acc_features_base = acc_features(acc_wrist_baseline)

        np.reshape(eda_features_stress, (1,-1))

        #print(eda_features_stress.shape, temp_features_stress.shape,ecg_features_time_stress.shape, ecg_features_freq_stress.shape)

<<<<<<< Updated upstream
        features_stress = np.vstack((features_stress, np.hstack((eda_features_stress, temp_features_stress, ecg_features_time_stress, ecg_features_freq_stress, emg_features_stress))))
        features_base = np.vstack((features_base, np.hstack((eda_features_base, temp_features_base, ecg_features_time_base, ecg_features_freq_base, emg_features_base))))
=======
        features_stress = np.vstack((features_stress, np.hstack((eda_features_stress, temp_features_stress, ecg_features_time_stress, ecg_features_freq_stress, emg_features_stress, acc_features_stress))  ))
        features_base = np.vstack((features_base, np.hstack((eda_features_base, temp_features_base, ecg_features_time_base, ecg_features_freq_base, emg_features_base, acc_features_base)) ))
>>>>>>> Stashed changes

    features_base = features_base[1:,:]
    features_stress = features_stress[1:,:]

    print(features_base)
    print(features_stress)
    
    features_in = np.vstack((features_base,features_stress))
    stress_state = np.append( np.zeros(features_base.shape[0]) , np.ones(features_stress.shape[0]) )

    return(features_in, stress_state)
    

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

feat_test, stress_test = extraction(test)
feat_train, stress_train = extraction(train)

print("feat_train shape=", feat_train.size, '\n', "stress_train shape", stress_train.size)
print("feat_test shape=", feat_test.size, '\n', "stress_test shape", stress_test.size)

def saving(feat, stress_state):
    store_feat = pd.DataFrame(feat)
    store_stress = pd.DataFrame(stress_state)

    table_feat = pa.Table.from_pandas(store_feat, preserve_index=False)
    table_stress = pa.Table.from_pandas(store_stress, preserve_index=False)

    
    if(feat.size>4000):
        print("if")
        pq.write_table(table_feat, 'Feature_train.parquet')
        pq.write_table(table_stress, 'Stress_train.parquet')

    else:
        print("else")
        pq.write_table(table_feat, 'Feature_test.parquet')
        pq.write_table(table_stress, 'Stress_test.parquet')


    return 

saving(feat_test, stress_test)
saving(feat_train, stress_train)

#print("type after saving", type(feat_train))

