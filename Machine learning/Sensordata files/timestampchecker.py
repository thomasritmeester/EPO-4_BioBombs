import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

inputfile = 'ECGdata.csv'
importData = pd.read_csv(inputfile)
print(importData)
timestamps = importData["TimeStamp"]

for i in range(1, len(timestamps)):
    if (timestamps[i] - timestamps[i-1]) < 0:
        print("jump at " + str(timestamps[i-1]))