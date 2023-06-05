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

from sklearn.utils import shuffle
pd_feat, pd_stress = shuffle(pd_feat, pd_stress)

X_train=pd_feat[:845]
X_test=pd_feat[845:]
y_train=pd_stress[:845]
y_test=pd_stress[845:]
y_train=y_train.to_numpy()
y_test=y_test.to_numpy()
y_train=y_train.ravel()
y_test=y_test.ravel()



print(pd_feat.shape, pd_stress.shape)
# print(np_feat.shape, np_stress.shape)
#X_train, X_test, y_train, y_test = train_test_split(pd_feat, pd_stress, test_size=0.25, random_state=42)




########################################################################
#LDA
lda=LDA(n_components=1)
train_lda=lda.fit(X_train, y_train)
test_lda=lda.predict(X_test)

# print(test_lda.shape)
# print(y_test.shape)

score= lda.score(X_test,y_test)
print('Score:', score)
print('\n')

########################################################################
#Neural Network

#Best Batchsize= 16
#Best Hidden layer nodes= approx 16,32,32
#Best droppout= 0.19
#Best optimizer = Nadam
#Best loss = "binary_crossentropy"

#print()
#for j in range (12):

#dropoutje=0.1

# input_nodes = X_train.shape[1]
# hidden_layer_1_nodes = 16
# hidden_layer_2_nodes = 32
# hidden_layer_3_nodes = 32
# output_layer = 1

# # initializing a sequential model
# full_model = Sequential()

# # adding layers
# full_model.add(Dense(hidden_layer_1_nodes,input_dim=input_nodes , activation='relu'))
# #full_model.add(Dropout(dropoutje))
# full_model.add(Dense(hidden_layer_2_nodes, activation='relu'))
# #full_model.add(Dropout(dropoutje))
# full_model.add(Dense(hidden_layer_3_nodes, activation='relu'))
# #full_model.add(Dropout(dropoutje))
# full_model.add(Dense(output_layer, activation='sigmoid'))

# #full_model.summary()

# # Compiling the DNN
# full_model.compile(optimizer="Nadam", loss="binary_crossentropy", metrics=['accuracy'])

# history = full_model.fit(X_train,y_train,validation_data=(X_test,y_test), epochs=512, batch_size=(16), verbose=0)	

# # req1=np.mean(history.history['val_accuracy'])>np.mean(history.history['accuracy'])
# # req2=np.max(history.history['val_accuracy'])>np.max(history.history['accuracy'])
# #req3=np.max(history.history['val_accuracy'])>0.96
# print("This is the accuracy:", np.max(history.history['accuracy']))
# #print(j)
# #if (req1==True and req2==True):     # and req3==True
# #full_model.save(f"Best NN {j}")
# plt.figure()
# plt.plot(history.history['accuracy']) #label = f"training{dropoutje}")
# plt.plot(history.history['val_accuracy'])# label = f"test{dropoutje}")
# print("This is the validation, not overfitted:", np.max(history.history['val_accuracy']))
# plt.legend()


new_model = keras.models.load_model('Best NN 8')
new_model.summary()
# Evaluate the restored model
loss, acc = new_model.evaluate(X_test, y_test, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))
