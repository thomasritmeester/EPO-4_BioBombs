# %reload_ext autoreload
# %autoreload 2


# import os
import pickle
import numpy as np
# from scipy import signal
# from scipy.signal import butter, iirnotch, lfilter
# import numpy as np
import matplotlib.pyplot as plt

from os import listdir
from os.path import isfile, join, isdir
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import scipy.ndimage
from RESP import RESPprep 
from EDA import EDAprep

#Extracting the data from the 
data_set_path = "/Users/kakis/OneDrive/Documenten/unie/Delft/BSc Electrical Engineering/Year 2/EPO/EPO-4/WESAD/WESAD/"
subject = ["S2",'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S13', 'S14', 'S15', 'S16', 'S17']

class read_data_of_one_subject:
    """Read data from WESAD dataset"""
    def __init__(self, path, subject):
        self.keys = ['label', 'subject', 'signal']
        self.signal_keys = ['wrist', 'chest']
        self.chest_sensor_keys = ['ACC', 'ECG', 'EDA', 'EMG', 'Resp', 'Temp']
        self.wrist_sensor_keys = ['ACC', 'BVP', 'EDA', 'TEMP']
        #os.chdir(path)
        #os.chdir(subject)
        with open(path + subject +'/'+subject + '.pkl', 'rb') as file:
            data = pickle.load(file, encoding='latin1')
        self.data = data

    def get_labels(self):
        return self.data[self.keys[0]]

    def get_wrist_data(self):
        """"""
        #label = self.data[self.keys[0]]
        assert subject == self.data[self.keys[1]]
        signal = self.data[self.keys[2]]
        wrist_data = signal[self.signal_keys[0]]
        #wrist_ACC = wrist_data[self.wrist_sensor_keys[0]]
        #wrist_ECG = wrist_data[self.wrist_sensor_keys[1]]
        return wrist_data

    def get_chest_data(self):
        """"""
        signal = self.data[self.keys[2]]
        chest_data = signal[self.signal_keys[1]]
        return chest_data
    

print(len(subject))
fs = 700

i = 2
obj_data = {}

obj_data[subject[i]] = read_data_of_one_subject(data_set_path, subject[i])
# print(obj_data[subject[i]].data)
chest_data_dict = obj_data[subject[i]].get_chest_data()

labels = obj_data[subject[i]].get_labels()
baseline = np.asarray([idx for idx, val in enumerate(labels) if val == 1])
stress = np.asarray([idx for idx, val in enumerate(labels) if val == 2])

resp_stress = chest_data_dict['Resp'][stress, 0]
resp_base = chest_data_dict['Resp'][baseline, 0]


RESP = []
# plt.plot(resp_base[:700*60])
RESP = RESPprep(700, resp_base, "stress")
RESP.alles(True)
# resp_filt_base = RESP.filtering_data_low()
#resp_filt_base = RESP.filtering_data_high(resp_filt_base)
#resp_filt_base = RESP.smoothing_data(resp_filt_base)


