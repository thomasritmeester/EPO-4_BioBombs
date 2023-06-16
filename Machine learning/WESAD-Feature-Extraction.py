import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
from scipy.signal import chirp, find_peaks, peak_widths, peak_prominences
from scipy.signal import chirp, find_peaks, peak_widths, peak_prominences
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from EDA_Features2 import *
from TEMP import *
from ECG_features_time import * 
from ECG_features_freq import *
from EMG_Features import *
from Remove_Movement import *
from Resp_Features import *
from wesad import read_data_of_one_subject
import warnings

from sklearn.utils import shuffle

warnings.simplefilter(action='ignore', category=FutureWarning)

#######################################################################
#Initializing the file, then making random train and test sets.
print("Start!")

data_set_path = "WESAD/"
subject = ["S2",'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S13', 'S14', 'S15', 'S16', 'S17']
gender_array= [ '2',   '2', '2', '2', '-2', '2', '-2', '2', '2', '-2', '2', '2', '2', '2', '-2']
fs = 700



#features_base = np.asarray(np.zeros(77), dtype = "float")
#features_stress = np.asarray(np.zeros(77), dtype = "float")


        

all_data_df = pd.DataFrame()

for i in range(len(subject)): 
    patient_df = pd.DataFrame()

    print("Subject: ", subject[i])

    base_dict = {}
    stress_dict = {}
    obj_data = {}

    obj_data[subject[i]] = read_data_of_one_subject(data_set_path, subject[i])
    #print(obj_data[subject[i]].data)
    chest_data_dict = obj_data[subject[i]].get_chest_data()

    labels = obj_data[subject[i]].get_labels() 
    baseline = np.asarray([idx for idx,val in enumerate(labels) if val == 1])
    stress = np.asarray([idx for idx,val in enumerate(labels) if val == 2])


    ecg_data_stress=chest_data_dict['ECG'][stress,0]
    ecg_data_base=chest_data_dict['ECG'][baseline,0][int(60*700):]
    
    acc_chest_stress=chest_data_dict['ACC'][stress]
    acc_chest_baseline=chest_data_dict['ACC'][baseline][int(60*700):]

    eda_data_stress=chest_data_dict['EDA'][stress,0]
    eda_data_base=chest_data_dict['EDA'][baseline,0][int(60*700):]

    emg_data_stress=chest_data_dict['EMG'][stress,0]
    emg_data_base=chest_data_dict['EMG'][baseline,0][int(60*700):]
    
    temp_data_stress=chest_data_dict['Temp'][stress,0]
    temp_data_base=chest_data_dict['Temp'][baseline,0][int(60*700):]

    resp_data_stress=chest_data_dict['Resp'][stress,0]
    resp_data_base=chest_data_dict['Resp'][baseline,0][int(60*700):]


    print('ECG')
    ecg_features_time_base = ECG_time_data(ecg_data_base, 60)
    ecg_features_time_stress = ECG_time_data(ecg_data_stress, 60)
    
    print('EDA')
    eda_features_base = calc_eda_features(eda_data_base, 60)
    eda_features_stress = calc_eda_features(eda_data_stress, 60)
    
    print('EMG')
    emg_features_base = calc_emg_features(emg_data_base, 60)
    emg_features_stress = calc_emg_features(emg_data_stress, 60)

    print('TEMP')
    temp_features_base = calc_temp_features(temp_data_base, 60)
    temp_features_stress = calc_temp_features(temp_data_stress, 60)

    print('RESP')
    resp_features_base = calc_resp_features(resp_data_base, fs, 60)
    resp_features_stress = calc_resp_features(resp_data_stress, fs, 60)
    
    print('ECG')
    ecg_features_freq_base = ECG_freq_data(ecg_data_base, 60)
    ecg_features_freq_stress = ECG_freq_data(ecg_data_stress, 60)

    print('ACC')
    acc_features_stress = calc_acc_features(acc_chest_stress, 60)
    acc_features_base = calc_acc_features(acc_chest_baseline, 60)

    base_dict['EDA'] = eda_features_base
    base_dict['EMG'] = emg_features_base
    base_dict['TEMP'] = temp_features_base
    base_dict['ECG'] = pd.concat([ecg_features_time_base, ecg_features_freq_base], axis = 1)
    base_dict['ACC'] = acc_features_base
    base_dict['RESP'] = resp_features_base

    stress_dict['EDA'] = eda_features_stress
    stress_dict['EMG'] = emg_features_stress
    stress_dict['TEMP'] = temp_features_stress
    stress_dict['ECG'] = pd.concat([ecg_features_time_stress, ecg_features_freq_stress], axis = 1)
    stress_dict['ACC'] = acc_features_stress
    stress_dict['RESP'] = resp_features_stress


    patient_stress_df = pd.concat(stress_dict, axis = 1)
    patient_base_df = pd.concat(base_dict, axis = 1)


    stress_state = np.append(np.zeros(patient_base_df.shape[0]) , np.ones(patient_stress_df.shape[0]))
    stress_out = {}
    stress_df = pd.DataFrame(stress_state, columns = ['Out'])
    stress_out['Out'] = stress_df
    stress_df = pd.concat(stress_out, axis = 1)
    patient_df = pd.concat([patient_df, pd.concat([patient_base_df, patient_stress_df], ignore_index = True),stress_df], axis = 1)

    subject_list = np.asarray([subject[i]]*len(patient_df.index))
    subject_name = {}
    subject_df = pd.DataFrame(subject_list, columns = ['Subject'])
    subject_name['Subject'] = subject_df
    subject_name_df = pd.concat(subject_name, axis = 1)

    gender_list = np.asarray([gender_array[i]]*len(patient_df.index))
    gender = {}
    gender_temp = pd.DataFrame(gender_list, columns = ['Gender'])
    gender['Gender'] = gender_temp
    gender_df = pd.concat(gender, axis = 1)


    patient_df = pd.concat([subject_name_df, gender_df, patient_df], axis = 1)

    all_data_df = pd.concat([all_data_df, patient_df], ignore_index = True)
    




