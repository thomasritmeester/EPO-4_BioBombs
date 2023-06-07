import os
import pickle
import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import butter, iirnotch, lfilter
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft

from os import listdir
from os.path import isfile, join, isdir
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import scipy.ndimage
from ACC import ACCprep
from scipy import interpolate
from wesad import read_data_of_one_subject
from itertools import compress

data_set_path = "WESAD/"
subject = ["S2",'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S13', 'S14', 'S15', 'S16', 'S17']

#wrist sampling frequencies
fs_dict = {'ACC': 32, 'label': 700}



def remove_movement(chest_data_dict, subject_id, stress, baseline, baseline_signals, stress_signals):
    obj_data = {}
    fs = 700

    obj_data[subject[subject_id]] = read_data_of_one_subject(data_set_path, subject[subject_id])

    #print(obj_data[subject[i]].data)
    wrist_data_dict = obj_data[subject[subject_id]].get_wrist_data()

    labels = obj_data[subject[subject_id]].labels

    #Chest ACC data
    acc_chest_stress=chest_data_dict['ACC'][stress]
    acc_chest_baseline=chest_data_dict['ACC'][baseline]


    acc_df = pd.DataFrame(wrist_data_dict['ACC'], columns=['ACC_x', 'ACC_y', 'ACC_z'])
    acc_df.index =  [(1 / fs_dict['ACC']) * i for i in range(len(acc_df))]
    label_df = pd.DataFrame(labels, columns=['label'])
    label_df.index = [(1 / fs_dict['label']) * i for i in range(len(label_df))]

    df = pd.DataFrame()
    df = df.join(acc_df, how='outer')
    df = df.join(label_df, how='outer')
    df['label'] = df['label'].fillna(method='bfill')
    df.reset_index(drop=True, inplace=True)
    df['label'] = df['label'].fillna(method='bfill')
    df.reset_index(drop=True, inplace=True)
    df.dropna(inplace=True)

    grouped = df.groupby('label')
    baseline = grouped.get_group(1)
    stress = grouped.get_group(2)


    acc_wrist_base_short = baseline.to_numpy()[:,:3]
    acc_wrist_stress_short = baseline.to_numpy()[:,:3]

    
    

    acc_wrist_stress = np.zeros([len(acc_chest_stress), 3])
    acc_wrist_baseline = np.zeros([len(acc_chest_baseline), 3])


    #Changes the size of the wirst array to match the length of the chest array
    wrist_arrays = [acc_wrist_stress_short,  acc_wrist_base_short, acc_wrist_stress,  acc_wrist_baseline, acc_chest_stress, acc_chest_baseline]

    for j in range(0,2):
        for i in range(np.shape(wrist_arrays[j+2])[1]):
            x = np.arange(0, len(wrist_arrays[j][0:,i]), 1)
            f = interpolate.interp1d(x, wrist_arrays[j][0:,i], bounds_error=False, fill_value=0)
            xnew = np.linspace(0, len(wrist_arrays[j][0:,i]), len(wrist_arrays[j+4]))
            ynew = f(xnew)
            wrist_arrays[j+2][:,i] = ynew



    ACC = ACCprep(fs,np.linspace(0,len(acc_wrist_stress)), "")

    #Create the indices of 0/1 on which data to exclude
    total_indices_stress = np.full(np.shape(acc_wrist_stress)[0], True)
    total_indices_baseline = np.full(np.shape(acc_wrist_baseline)[0], True)


    #Combine the wrist acc for both stress and baseline
    filtering_arrays = [acc_wrist_stress, acc_wrist_baseline, total_indices_stress, total_indices_baseline]

    for k in range(0,2):
        for j in range( np.shape(filtering_arrays[k])[1]):
            acc_filt= ACC.filtering_data(filtering_arrays[k][:,j])

            threshold_up = np.mean(acc_filt) + 2.7*np.std(acc_filt)
            threshold_down = np.mean(acc_filt) - 2.7*np.std(acc_filt)

            indices = (np.logical_or((acc_filt > threshold_up),(acc_filt < threshold_down))).astype(int)

            fixed_indices = np.full(np.shape(indices), False)
            thresh = 18*fs # 9 second on either side
            for i in range(int(thresh/2), int(len(indices)-(thresh/2))):
                if( int(np.sum(indices[int(i-(thresh/2)):int(i+(thresh/2))])) != 0):
                    fixed_indices[i] = False
                else:
                    fixed_indices[i] = True

            filtering_arrays[k+2] = np.logical_and(filtering_arrays[k+2], fixed_indices)

            # plt.figure(figsize = (12,8))
            # plt.plot(acc_filt)
            # plt.plot(threshold_up*~(fixed_indices))
            # plt.plot(threshold_up*~(filtering_arrays[k+2]), color = 'red')
            # plt.show()

    # total_indices_stress = filtering_arrays[2]
    # total_indices_baseline = filtering_arrays[3]
    # plt.figure(figsize = (12,8))
    # plt.plot(total_indices_stress, color = 'purple')
    # plt.plot(total_indices_baseline, color = 'orange')
    # plt.show()


    for i in range(len(baseline_signals)):
        baseline_signals[i] = baseline_signals[i][:len(total_indices_baseline)]
        baseline_signals[i] = np.asarray(list(compress(baseline_signals[i], total_indices_baseline)))
        
    for i in range(len(stress_signals)):
        stress_signals[i] = stress_signals[i][:len(total_indices_stress)]
        stress_signals[i] = np.asarray(list(compress(stress_signals[i], total_indices_stress)))

    return baseline_signals[0], baseline_signals[1], baseline_signals[2], stress_signals[0], stress_signals[1], stress_signals[2], acc_wrist_stress, acc_wrist_baseline

    

def acc_features(acc_wrist):
    fs = 700
    window = int(0.5*60*fs)

    #acc_features = np.zeros(15)

    acc_features = pd.DataFrame()

    
    acc = acc_wrist[:,0]
    t_tot = (len(acc)//(int(window)))

    merged_array = np.zeros((t_tot, window))
    for j in range(np.shape(acc_wrist)[1]):
        
        acc = acc_wrist[:,j]
        t_tot = (len(acc)//(int(window)))
        two_d_array = np.zeros((window))

        for i in range(t_tot):
            acc_data = acc[i*int(window):(i+1)*int(window)]
            two_d_array = np.vstack([two_d_array, acc_data])

        two_d_array = two_d_array[1:,:]    
        merged_array = np.dstack((merged_array,two_d_array))
    
    merged_array = merged_array.swapaxes(0,2).swapaxes(1,2)
    merged_array = merged_array[1:,:,:]

    for n in range(t_tot):
        temp_array = np.asarray(np.asarray([], dtype = "float"))
        for m in range(np.shape(acc_wrist)[1]):
            #loops through x, y and z                    
            temp_array = np.append(temp_array,np.mean(merged_array[m][n,:]))
            temp_array = np.append(temp_array,np.std(merged_array[m,:,n]))
            temp_array = np.append(temp_array,np.sum(np.abs(merged_array[m,:,n])))

            acc_norm = (merged_array[m,:,n] - np.mean(merged_array[m,:,n]))
            Y = np.abs(fft(acc_norm))
            F = np.arange(0,fs,fs/np.size(Y))
            F = F[:len(Y)]
            temp_array = np.append(temp_array,F[np.argmax(Y)])

        #mean  of the std, sum and abs of all three axes
        temp_array = np.append(temp_array, np.mean(temp_array[0:3]))
        temp_array = np.append(temp_array, np.mean(temp_array[3:6]))
        temp_array = np.append(temp_array, np.mean(temp_array[6:9]))


        temp_pd = pd.DataFrame([temp_array], columns = ['mean_x', 'mean_y', 'mean_z', 'std_x', 'std_y', 'std_z', 'integral_x', 'integral_y', 'integral_z', 'Max freq_x','Max freq_y', 'Max freq_z' , 'mean_3D', 'std 3D', 'integral 3D'])
        
        acc_features = pd.concat([acc_features, temp_pd], ignore_index=True)
        
        #acc_features = np.vstack((acc_features,temp_array))
            
    #acc_features  = acc_features[1:,:] 

    return acc_features

