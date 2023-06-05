#NN test file
#####################################################################
#Importing Packages
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import ModelCheckpoint
import keras as keras
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA




######################################################################
#Reading in the features and making a Train/test split
Features_in = pq.read_table('Feature.parquet')
Stress_state = pq.read_table('Stress.parquet')

pd_feat= Features_in.to_pandas()
pd_stress= Stress_state.to_pandas()

# print(np_feat.shape, np_stress.shape)
X_train, X_test, y_train, y_test = train_test_split(pd_feat, pd_stress, test_size=0.25, random_state=42)




########################################################################
#LDA
lda=LDA(n_components=1)
train_lda=lda.fit(X_train, y_train)
test_lda=lda.predict(X_test)

# print(test_lda.shape)
# print(y_test.shape)

score= lda.score(X_test,y_test)
print('Score:', score)




########################################################################
#Neural Network
input_nodes = X_train.shape[1]
hidden_layer_1_nodes = 64
hidden_layer_2_nodes = 128
hidden_layer_3_nodes = 64
output_layer = 1

# initializing a sequential model
full_model = Sequential()

# adding layers
full_model.add(Dense(hidden_layer_1_nodes,input_dim=input_nodes , activation='relu'))
full_model.add(Dropout(0.25))
full_model.add(Dense(hidden_layer_2_nodes, activation='relu'))
full_model.add(Dropout(0.25))
full_model.add(Dense(hidden_layer_3_nodes, activation='relu'))
full_model.add(Dropout(0.25))
full_model.add(Dense(output_layer, activation='sigmoid'))

full_model.summary()

# Compiling the ANN
full_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = full_model.fit(X_train,y_train,validation_data=(X_test,y_test), epochs=512, batch_size=32, verbose=2)	

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
print("best accuracy:",np.max(history.history['accuracy']))
print("best val-accuracy:", np.max(history.history['val_accuracy']))
