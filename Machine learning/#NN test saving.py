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


Features_in = pq.read_table('Feature.parquet')
Stress_state = pq.read_table('Stress.parquet')

pd_feat= Features_in.to_pandas()
pd_stress= Stress_state.to_pandas()

X_train, X_test, y_train, y_test = train_test_split(pd_feat, pd_stress, test_size=0.25, random_state=42)

new_model = keras.models.load_model('Best NN')
new_model.summary()
# Evaluate the restored model
loss, acc = new_model.evaluate(X_test, y_test, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))
plt.plot(loss, label = "test")