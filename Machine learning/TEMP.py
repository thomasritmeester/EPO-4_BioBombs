import numpy as np
import pandas as pd

def calc_temp_features(data, frame):

  SamplingRate = 700
  TimeWindow = frame
  WindowSize = int(SamplingRate*TimeWindow)

  features = np.zeros(6)

  #temp_features = np.asarray(np.zeros(6), dtype = "float")
  temp_features = pd.DataFrame()


  t_tot = int(len(data)/WindowSize)
  for i in range(t_tot):
    data_window = data[WindowSize*i:WindowSize+WindowSize*i]
    mean = np.mean(data_window)
    std = np.std(data_window)
    minimum = np.amin(data_window)
    maximum = np.amax(data_window)
    d_range = maximum/minimum
    slope = np.polyfit(np.arange(0,WindowSize,1),data_window,1)[0]
    features[0] = mean
    features[1] = std
    features[2] = minimum
    features[3] = maximum
    features[4] = slope
    features[5] = d_range


    pd_features = pd.DataFrame([features], columns = ['mean', 'std', 'minimum', 'maximum', 'slope', 'Dynamic Range'])

    temp_features = pd.concat([temp_features, pd_features], ignore_index=True)
    #temp_features = np.vstack((temp_features,features))

  return temp_features
  
    
