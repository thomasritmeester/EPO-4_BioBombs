import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
from scipy.signal import chirp, find_peaks, peak_widths, peak_prominences
from scipy.signal import chirp, find_peaks, peak_widths, peak_prominences
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from EDA_Features2 import *
from TEMP import *
from ECG_features2 import * 
from ECG_features3 import * 
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

print("Start!")

data_set_path = "D:/Downloads/WESAD/WESAD/"
subject = ["S2",'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S13', 'S14', 'S15', 'S16', 'S17']

features_base = np.asarray(np.zeros(36), dtype = "float")
features_stress = np.asarray(np.zeros(36), dtype = "float")

#######################################################################
#Reading out subjects and calling the feature extraction functions
for i in range(len(subject)):     
    print("subject: ", subject[i])

    obj_data = {}

    obj_data[subject[i]] = read_data_of_one_subject(data_set_path, subject[i])
    #print(obj_data[subject[i]].data)
    chest_data_dict = obj_data[subject[i]].get_chest_data()

    labels = obj_data[subject[i]].get_labels() 
    baseline = np.asarray([idx for idx,val in enumerate(labels) if val == 1])
    stress = np.asarray([idx for idx,val in enumerate(labels) if val == 2])

    eda_data_stress=chest_data_dict['EDA'][stress,0]
    eda_data_base=chest_data_dict['EDA'][baseline,0]

    temp_data_stress=chest_data_dict['Temp'][stress,0]
    temp_data_base=chest_data_dict['Temp'][baseline,0]

    ecg_data_stress=chest_data_dict['ECG'][stress,0]
    ecg_data_base=chest_data_dict['ECG'][baseline,0]    

    eda_features_base = calc_eda_features(eda_data_base)
    eda_features_stress = calc_eda_features(eda_data_stress)

    temp_features_base = calc_temp_features(temp_data_base)
    temp_features_stress = calc_temp_features(temp_data_stress)

    ecg_features_time_base = ECG_time_data(ecg_data_base)
    ecg_features_time_stress = ECG_time_data(ecg_data_stress)

    ecg_features_freq_base = ECG_freq_data(ecg_data_base)
    ecg_features_freq_stress = ECG_freq_data(ecg_data_stress)

    #print(ecg_features_freq_base)
    #print(ecg_features_freq_base.shape)

    #print(ecg_features_freq_stress)
    #print(ecg_features_freq_stress.shape)

    np.reshape(eda_features_stress, (1,-1))

    #print(eda_features_stress.shape, temp_features_stress.shape,ecg_features_time_stress.shape, ecg_features_freq_stress.shape)

    features_stress = np.vstack((features_stress, np.hstack((eda_features_stress, temp_features_stress, ecg_features_time_stress, ecg_features_freq_stress))  ))
    features_base = np.vstack((features_base, np.hstack((eda_features_base, temp_features_base, ecg_features_time_base, ecg_features_freq_base)) ))

features_base = features_base[1:,:]
features_stress = features_stress[1:,:]
#print("feat_base:")
#print(features_base)
#print(features_base.shape)
#print("feat_stress:")
#print(features_stress)
#print(features_stress.shape)

features_in = np.vstack((features_base,features_stress))
#print("feat_in:")
#print(features_in)
#print(features_in.shape)
stress_state = np.append( np.zeros(features_base.shape[0]) , np.ones(features_stress.shape[0]) )
#print("stress_state:")
#print(stress_state)
#print(stress_state.shape)
#stress_state = np.ravel(stress_state)


########################################################################
#LDA
X_train, X_test, y_train, y_test = train_test_split(features_in, stress_state, test_size=0.25, random_state=42)

lda=LDA(n_components=1)
train_lda=lda.fit(X_train, y_train)
test_lda=lda.predict(X_test)

# print(test_lda.shape)
# print(y_test.shape)

score= lda.score(X_test,y_test)
print('Score:', score)




######################################################################
#Neural Network
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import ModelCheckpoint
import keras as keras

# # Create simple Neural Network model
input_nodes = X_train.shape[1]
hidden_layer_1_nodes = 1
hidden_layer_2_nodes = 128
hidden_layer_3_nodes = 256
output_layer = 1

# initializing a sequential model
full_model = Sequential()

# adding layers
full_model.add(Dense(hidden_layer_1_nodes, activation='relu'))
full_model.add(Dropout(0.25))
full_model.add(Dense(hidden_layer_2_nodes, activation='relu'))
full_model.add(Dropout(0.25))
full_model.add(Dense(hidden_layer_3_nodes, activation='relu'))

full_model.add(Dense(output_layer, activation='relu'))

# Compiling the DNN
full_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
full_model.summary()

checkpoint_name = 'Weights-{epoch:03d}--{val_accuracy:.5f}.hdf5' 
checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_accuracy', verbose = 0, save_best_only = True, mode ='auto')
callbacks_list = [checkpoint]

history = full_model.fit(X_train,y_train,validation_data=(X_test,y_test), epochs=500, batch_size=32, verbose=1)	

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
print("best accuracy:",np.max(history.history['accuracy']))
print("best val-accuracy:", np.max(history.history['val_accuracy']))

#######################################################################
## K Cross fold validation

from numpy import mean
from numpy import std
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# prepare the cross-validation procedure
cv = KFold(n_splits=5, shuffle=True)

# evaluate model
scores = cross_val_score(lda, features_in, stress_state, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
print(scores)