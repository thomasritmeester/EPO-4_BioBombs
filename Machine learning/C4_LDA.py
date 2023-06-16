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
from CrossFold import *
from wesad import read_data_of_one_subject
import warnings

from sklearn.utils import shuffle

warnings.simplefilter(action='ignore', category=FutureWarning)

#######################################################################
#Initializing the file
print("Start!")

base_dict = {}
stress_dict = {}

subjects=["P1","P2","P5","P6","P9","P10","P11","P12"] #,"P6", P3 later to be added
gender_array= [ '2',   '2', '2', '2', '-2', '2', '-2', '2', '2', '-2', '2', '2', '2', '2', '-2'] 

########################################################################
# Making a random shuffle of subjects:
     
# sub_shuf = shuffle(subjects)
# print(sub_shuf)
# train=sub_shuf[:len(subjects)-1]
# test=[sub_shuf[-1]]  
# print("Test subject is:", test[0])

########################################################################
#Function to call for the train and the test subject.

def extraction (train_test):
     
    all_data_df = pd.DataFrame()   

    #Use if crossfold validation is not used:
    # features_base = pd.DataFrame()
    # features_stress = pd.DataFrame()
    # features_in_df = pd.DataFrame()

    for i in range (len(train_test)): 
        patient_df = pd.DataFrame()

        print("Subject: ", train_test[i])

        base_data = pd.read_csv(f"D:\Documents\GitHub\EPO-4_BioBombs\Machine learning\Sensordata files\C4 raw data\Best Data\{train_test[i]} baseline.csv")
        stress_data = pd.read_csv(f"D:\Documents\GitHub\EPO-4_BioBombs\Machine learning\Sensordata files\C4 raw data\Best Data\{train_test[i]} stress.csv")

        ecg_data_base=base_data['ECG']
        ecg_data_stress=stress_data['ECG']

        ecg_data_base=ecg_data_base.to_numpy()
        ecg_features_time_base = ECG_time_data(ecg_data_base)
        ecg_features_freq_base = ECG_freq_data(ecg_data_base)

        acc_chest_stress=stress_data['ACC_X', 'ACC_Y', 'ACC_Z']

        eda_data_stress=stress_data['EDA']
        eda_data_base=base_data['EDA']

        emg_data_stress=stress_data['EMG']
        emg_data_base=base_data['EMG']

        temp_data_stress=stress_data['Temp']
        temp_data_base=base_data['Temp']
        
        #Only use if acc is used:
        #Signals to be processed by ACC
        baseline_signals = [eda_data_base, emg_data_base, ecg_data_base]
        stress_signals = [eda_data_stress, emg_data_stress, ecg_data_stress]
        #eda_data_base, emg_data_base, ecg_data_base, eda_data_stress, emg_data_stress, ecg_data_stress, acc_wrist_stress, acc_wrist_baseline = remove_movement(sensor_data, i, stress, baseline, baseline_signals, stress_signals)

        eda_features_base = calc_eda_features(eda_data_base)
        eda_features_stress = calc_eda_features(eda_data_stress)

        emg_features_base = calc_emg_features(emg_data_base)
        emg_features_stress = calc_emg_features(emg_data_stress)

        temp_features_base = calc_temp_features(temp_data_base)
        temp_features_stress = calc_temp_features(temp_data_stress)

        ecg_features_time_stress = ECG_time_data(ecg_data_stress)
        ecg_features_freq_stress = ECG_freq_data(ecg_data_stress)

        acc_features_base=calc_acc_features
        acc_features_stress=calc_acc_features

        base_dict['EDA'] = eda_features_base
        base_dict['EMG'] = emg_features_base
        base_dict['TEMP'] = temp_features_base
        base_dict['ECG'] = pd.concat([ecg_features_time_base, ecg_features_freq_base], axis = 1)

        base_dict['ACC'] = acc_features_base

        stress_dict['EDA'] = eda_features_stress
        stress_dict['EMG'] = emg_features_stress
        stress_dict['TEMP'] = temp_features_stress
        stress_dict['ECG'] = pd.concat([ecg_features_time_stress, ecg_features_freq_stress], axis = 1)
        stress_dict['ACC'] = acc_features_stress

        
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


        #Use if not all data is used:

        # features_stress = pd.concat([features_stress, pd.concat(stress_dict, axis = 1)], ignore_index = True)
        # features_base = pd.concat([features_base, pd.concat(base_dict, axis = 1)], ignore_index = True)

        # features_in_df = pd.concat([features_base, features_stress], ignore_index = True)
        # features_in = features_in_df.to_numpy()

        # stress_state = np.append(np.zeros(features_base.shape[0]) , np.ones(features_stress.shape[0]))

        # stress_out = {}
        # stress_df = pd.DataFrame(stress_state, columns = ['Out'])
        # stress_out['Out'] = stress_df
        # stress_df = pd.concat(stress_out, axis = 1)
        # everything_all = pd.concat([features_in_df, stress_df], axis = 1)
        # everything_all.to_csv('all_data.csv')

    return all_data_df      #features_in, stress_state,

# print('\n',"First the training subjects")
# feat_train, stress_train,_ = extraction(train)
# print('\n',"Then the test subject")
# feat_test, stress_test,_ = extraction(test)

all_data_df=extraction(subjects)

lda=LDA(n_components=1)
accuracy_score, f1_score, mcc_score=CrossFold(subjects, lda, all_data_df, data = 'False')
print("accuracy_score=", accuracy_score, '\n')
print("f1_score=", accuracy_score, '\n')
print("mcc_score", accuracy_score, '\n')

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# print('\n',"feat_train shape=", feat_train.size, '\n',"stress_train shape", stress_train.size)
# print('\n',"feat_test shape=", feat_test.size, '\n',"stress_test shape", stress_test.size)

# size=feat_test.size
# print(feat_test.size)

# def saving(feat, stress_state):
#     store_feat = pd.DataFrame(feat)
#     store_stress = pd.DataFrame(stress_state)

#     table_feat = pa.Table.from_pandas(store_feat, preserve_index=False)
#     table_stress = pa.Table.from_pandas(store_stress, preserve_index=False)


#     if(feat.size>3000):
#         print("if")
#         pq.write_table(table_feat, 'Feature_train.parquet')
#         pq.write_table(table_stress, 'Stress_train.parquet')

#     else:
#         print("else")
#         pq.write_table(table_feat, 'Feature_test.parquet')
#         pq.write_table(table_stress, 'Stress_test.parquet')


#     return 

# #print("stress_df",stress_df, stress_df.shape)
# #print(features_base.shape, features_stress.shape)
# saving(feat_test, stress_test)
# saving(feat_train, stress_train)

# #X_train, X_test, y_train, y_test = train_test_split(features_in, stress_df, test_size=0.33, random_state=42, shuffle=True)
# #feat_train
# # X_test =feat_test
# # y_test =stress_test
# # X_train =feat_train
# # y_train =stress_train

# # lda=LDA(n_components=1)
# # train_lda=lda.fit(X_train, y_train)
# # test_lda=lda.predict(X_test) 
# # score= lda.score(X_test,y_test)
# # print("Score no CV=", score, "\n")


# # # K Cross fold validation

# from numpy import mean
# from numpy import std
# from sklearn.datasets import make_classification
# from sklearn.model_selection import KFold
# from sklearn.model_selection import cross_val_score
# from sklearn.linear_model import LogisticRegression



# # prepare the cross-validation procedure
# cv = KFold(n_splits=5, shuffle=True)

# # evaluate model
# scores = cross_val_score(lda, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
# # report performance
# print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
# print(scores)

