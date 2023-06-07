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

warnings.simplefilter(action='ignore', category=FutureWarning)

print("Start!")

data_set_path = "WESAD/" ##CHANGE DEPENDING ON FOLDER LOCATION
subject = ["S2",'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S13', 'S14', 'S15', 'S16', 'S17']

#features_base = np.asarray(np.zeros(77), dtype = "float")
#features_stress = np.asarray(np.zeros(77), dtype = "float")

features_base = pd.DataFrame()
features_stress = pd.DataFrame()
features_in_df = pd.DataFrame()

for i in range(len(subject)):

    base_dict = {}
    stress_dict = {}
    obj_data = {}
    print(subject[i])

    obj_data[subject[i]] = read_data_of_one_subject(data_set_path, subject[i])
    #print(obj_data[subject[i]].data)
    chest_data_dict = obj_data[subject[i]].get_chest_data()

    labels = obj_data[subject[i]].get_labels() 
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

    #Signals to be processed by ACC
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

    features_stress = pd.concat([features_stress, pd.concat(stress_dict, axis = 1)], ignore_index = True)
    features_base = pd.concat([features_base, pd.concat(base_dict, axis = 1)], ignore_index = True)


features_in_df = pd.concat([features_base, features_stress], ignore_index = True)
features_in = features_in_df.to_numpy()


stress_state = np.append(np.zeros(features_base.shape[0]) , np.ones(features_stress.shape[0]))

stress_out = {}
stress_df = pd.DataFrame(stress_state, columns = ['Out'])
stress_out['Out'] = stress_df
stress_df = pd.concat(stress_out, axis = 1)
everything_all = pd.concat([features_in_df, stress_df], axis = 1)
everything_all.to_csv('all_data.csv')



print(np.shape(features_in))

#X_train, X_test, y_train, y_test = train_test_split(features_in, stress_state, test_size=0.33, random_state=42)

lda=LDA(n_components=1)
#train_lda=lda.fit(X_train, y_train)
#test_lda=lda.predict(X_test)

#print(test_lda.shape)
#print(y_test.shape)

#score= lda.score(X_test,y_test)
#print(score)

# K Cross fold validation

from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression



# prepare the cross-validation procedure
cv = KFold(n_splits=15, shuffle=False)

# evaluate model
scores = cross_val_score(lda, features_in, stress_state, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
print(scores)
