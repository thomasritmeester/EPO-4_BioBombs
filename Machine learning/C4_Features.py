#C4_Features
from glob import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
from scipy.signal import chirp, find_peaks, peak_widths, peak_prominences
from scipy.signal import chirp, find_peaks, peak_widths, peak_prominences
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from ECG_features_time import * 
from EDA_Features2 import *
from TEMP import *
from ECG_features_freq import *
from EMG_Features import *
from Remove_Movement import *
from Resp_Features import *
from wesad import read_data_of_one_subject
import warnings
import numpy as np
import pandas as pd
from RESP import RESPprep



C4_subjects = ["P1", "P2", "P5", "P9", "P10", "P11", "P12"]
C4_gender = ['2', '-2', '2', '2', '-2', '-2', '2']




# for frame in np.arange(30,125,5):
#     all_data_df = pd.DataFrame()

#     for item in C4_subjects:

#         index_in_list = C4_subjects.index(item)
#         EXT = item + "_*"
#         all_csv_files = [file for path, subdir, files in os.walk("C4-Processed") for file in glob(os.path.join(path, EXT))]
#         all_csv_files.sort()

#         print(item)
#         df_base = pd.read_csv(all_csv_files[0], header = ([0]))
#         df_stress = pd.read_csv(all_csv_files[1], header = ([0]))


#         patient_df = pd.DataFrame()
#         base_dict = {}
#         stress_dict = {}
#         obj_data = {}

#         ecg_base = df_base['ECG'].to_numpy()
#         ecg_base = ecg_base - np.mean(ecg_base)
#         emg_base = df_base['EMG'].to_numpy()
#         eda_base = df_base['EDA'].to_numpy()
#         temp_base = df_base['Temp'].to_numpy()
#         acc_base = df_base[['ACC_X', 'ACC_Y', 'ACC_Z']].to_numpy()
#         resp_base = df_base['RESP'].dropna().to_numpy()

#         ecg_stress = df_stress['ECG'].to_numpy()
#         ecg_stress = ecg_stress - np.mean(ecg_stress)
#         emg_stress = df_stress['EMG'].to_numpy()
#         eda_stress = df_stress['EDA'].to_numpy()
#         temp_stress = df_stress['Temp'].to_numpy()
#         acc_stress = df_stress[['ACC_X', 'ACC_Y', 'ACC_Z']].to_numpy()
#         resp_stress = df_stress['RESP'].dropna().to_numpy()


#         ecg_features_time_base = ECG_time_data(ecg_base, frame)
#         ecg_features_time_stress = ECG_time_data(ecg_stress, frame)


#         eda_features_base = calc_eda_features(eda_base, frame)
#         eda_features_stress = calc_eda_features(eda_stress, frame)

#         emg_features_base = calc_emg_features(emg_base, frame)
#         emg_features_stress = calc_emg_features(emg_stress, frame)

#         temp_features_base = calc_temp_features(temp_base, frame)
#         temp_features_stress = calc_temp_features(temp_stress, frame)

#         ecg_features_freq_base = ECG_freq_data(ecg_base, frame)
#         ecg_features_freq_stress = ECG_freq_data(ecg_stress, frame)

#         acc_features_base = calc_acc_features(acc_base, frame)
#         acc_features_stress = calc_acc_features(acc_stress, frame)

#         resp_features_base = calc_resp_features(resp_base, 100, frame)
#         resp_features_stress = calc_resp_features(resp_stress, 100, frame)

#         base_dict['EDA'] = eda_features_base
#         base_dict['EMG'] = emg_features_base
#         base_dict['TEMP'] = temp_features_base
#         base_dict['ECG'] = pd.concat([ecg_features_time_base, ecg_features_freq_base], axis = 1)
#         base_dict['ACC'] = acc_features_base
#         base_dict['RESP'] = resp_features_base

#         stress_dict['EDA'] = eda_features_stress
#         stress_dict['EMG'] = emg_features_stress
#         stress_dict['TEMP'] = temp_features_stress
#         stress_dict['ECG'] = pd.concat([ecg_features_time_stress, ecg_features_freq_stress], axis = 1)
#         stress_dict['ACC'] = acc_features_stress
#         stress_dict['RESP'] = resp_features_stress


#         patient_stress_df = pd.concat(stress_dict, axis = 1)
#         patient_base_df = pd.concat(base_dict, axis = 1)



#         stress_state = np.append(np.zeros(patient_base_df.shape[0]) , np.ones(patient_stress_df.shape[0]))
#         stress_out = {}
#         stress_df = pd.DataFrame(stress_state, columns = ['Out'])
#         stress_out['Out'] = stress_df
#         stress_df = pd.concat(stress_out, axis = 1)
#         patient_df = pd.concat([patient_df, pd.concat([patient_base_df, patient_stress_df], ignore_index = True),stress_df], axis = 1)


#         subject_list = np.asarray([item]*len(patient_df.index))
#         subject_name = {}
#         subject_df = pd.DataFrame(subject_list, columns = ['Subject'])
#         subject_name['Subject'] = subject_df
#         subject_name_df = pd.concat(subject_name, axis = 1)

#         gender_list = np.asarray([C4_gender[index_in_list]]*len(patient_df.index))
# #         gender = {}
# #         gender_temp = pd.DataFrame(gender_list, columns = ['Gender'])
# #         gender['Gender'] = gender_temp
# #         gender_df = pd.concat(gender, axis = 1)


# #         patient_df = pd.concat([subject_name_df, gender_df, patient_df], axis = 1)

# #         all_data_df = pd.concat([all_data_df, patient_df], ignore_index = True)
        


#     all_data_df.to_csv('C4-features-windows/C4_' + str(frame) + '_sec.csv')

wesad_data_df = pd.read_csv('WESAD_data_1_min.csv', header = ([0,1]), index_col=0)

# all_data_df = pd.read_csv('C4-features-windows/C4_' + str(frame) + '_sec.csv', header = ([0,1]), index_col=0)



# list5 = list(all_data_df.columns.values)
# list5.remove(('Subject', 'Subject'))
# list5.remove(('Out', 'Out'))
# list5.remove(('Gender', 'Gender'))

# for item in (C4_subjects):
#     for i in range(len(list5)):
#         c4_base_feature_mean = np.mean(all_data_df[ (all_data_df['Subject']['Subject'] == str(item)) & (all_data_df['Out']['Out'] == 0)][str(list(list5)[i][0])][str(list(list5)[i][1])])
#         wesad_base_feature_mean = np.mean(wesad_data_df[wesad_data_df['Out']['Out'] == 0][str(list(list5)[i][0])][str(list(list5)[i][1])])
#         scaling_factor = 1+ ((wesad_base_feature_mean - c4_base_feature_mean)/c4_base_feature_mean)
#         all_data_df.loc[(all_data_df['Subject']['Subject'] == str(item)), (str(list(list5)[i][0]), [str(list(list5)[i][1])])]*= scaling_factor

# all_data_df.to_csv('C4-features-windows/C4_' + str(frame) + '_sec.csv')

############################################
# # Attempt  
list10 = list(wesad_data_df.columns.values)  
for k in range(len(list10)) :   
    wesad_base_feature_mean = np.mean(wesad_data_df[wesad_data_df['Out']['Out'] == 0][str(list(list10)[k][0])][str(list(list10)[k][1])])
    print(wesad_base_feature_mean.shape)
# for item in (subject):
#     for i in range(len(list10)):
#         wesad_subject_base_mean = np.mean(wesad_data_df[ (wesad_data_df['Subject']['Subject'] == str(item)) & (wesad_data_df['Out']['Out'] == 0)][str(list(list10)[i][0])][str(list(list10)[i][1])])
#         scaling_factor = 1+ ((wesad_base_feature_mean -wesad_subject_base_mean)/wesad_subject_base_mean)
#         wesad_data_df.loc[(wesad_data_df['Subject']['Subject'] == str(item)), (str(list(list10)[i][0]), [str(list(list10)[i][1])])]*= scaling_factor
