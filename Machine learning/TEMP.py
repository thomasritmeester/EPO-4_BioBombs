import numpy as np

def calc_temp_features(data):

  SamplingRate = 700
  TimeWindow = 30
  WindowSize = SamplingRate*TimeWindow

  features = np.zeros(5)

  temp_features = np.asarray(np.zeros(5), dtype = "float")


  for i in range(int(data.size/WindowSize)):
    data_window = data[WindowSize*i:WindowSize+WindowSize*i]
    mean = np.mean(data_window)
    std = np.std(data_window)
    minimum = np.amin(data_window)
    maximum = np.amax(data_window)
    slope = np.polyfit(np.arange(0,WindowSize,1),data_window,1)[0]
    features[0] = mean
    features[1] = std
    features[2] = minimum
    features[3] = maximum
    features[4] = slope
    temp_features = np.vstack((temp_features,features))

  return temp_features[1:,:]
  
    
