import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

class plotting:
    wesad = False

"""The class below allows you to access all the recordings and labels in the dataset."""

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

    def get_chest_data(self):
        """"""
        signal = self.data[self.keys[2]]
        chest_data = signal[self.signal_keys[1]]
        return chest_data

class CSV:
    subject = ["P6","P10","P11","P12","P2","P1","P5","P9"]
    def __init__(self,filename=""):
        self.filename=filename
        
    def __call__(self,p,Fs=700):
        #read cvs file
        data = pd.read_csv(p+" baseline"+".csv")
        # print(data.shape)
        # print(data)
        
        if p =="P6":
            
            self.timestamp = data['TimeStamp'].to_numpy()

            #seperateting the acceleration acxises
            self.ACC1x_base = data['Acc1 X'].to_numpy()
            self.ACC1y_base = data['Acc1 Y'].to_numpy()
            self.ACC1z_base = data['Acc1 z'].to_numpy()
            self.ACC2x_base = data['Acc2 X'].to_numpy()
            self.ACC2y_base = data['Acc2 Y'].to_numpy()
            self.ACC2z_base = data['Acc2 z'].to_numpy()
            self.temp_base = data['Temperature'].to_numpy()
            # seperate file into EMG, ECG and EDA
            self.emg_base = data['EMG Data'].to_numpy()
            self.ecg_base = data['ECG Data'].to_numpy()
        else:
            self.ACCx_base = data['ACC_X'].to_numpy()
            self.ACCy_base = data['ACC_Y'].to_numpy()
            self.ACCz_base = data['ACC_Z'].to_numpy()
            self.temp_base = data['Temp'].to_numpy()
            self.eda_base = data['EDA'].to_numpy()
            self.emg_base = data['EMG'].to_numpy()
            self.ecg_base = data['ECG'].to_numpy()
            self.timestamp = np.arange(0,len(self.emg_base)/Fs,1/Fs)[:len(self.emg_base)]
            self.resp_base = data["RESP"].to_numpy()

            
        data = pd.read_csv(p+" stress"+".csv")
        # print(data.shape)
        # print(data)
        
        if p =="P6":
            self.ACC1x_stress = data['Acc1 X'].to_numpy()
            self.ACC1y_stress = data['Acc1 Y'].to_numpy()
            self.ACC1z_stress = data['Acc1 z'].to_numpy()
            self.ACC2x_stress = data['Acc2 X'].to_numpy()
            self.ACC2y_stress = data['Acc2 Y'].to_numpy()
            self.ACC2z_stress = data['Acc2 z'].to_numpy()
            self.emg_stress = data['EMG'].to_numpy()
            self.ecg_stress = data['ECG'].to_numpy()
            self.temp_stress = data["Temperature"].to_numpy()
            self.t_stress = data['TimeStamp'].to_numpy()
        else:
            self.ACCx_stress = data['ACC_X'].to_numpy()
            self.ACCy_stress = data['ACC_Y'].to_numpy()
            self.ACCz_stress = data['ACC_Z'].to_numpy()
            self.eda_stress = data['EDA'].to_numpy()
            self.emg_stress = data['EMG'].to_numpy()
            self.ecg_stress = data['ECG'].to_numpy()
            self.temp_stress = data["Temp"].to_numpy()
            self.t_stress = np.arange(0,len(self.emg_stress)/Fs,1/Fs)[:len(self.emg_stress)]
            self.resp_stress = data["RESP"].to_numpy()

class WESAD:
    Fs= 700
    subject = ["S2","S3","S4","S5","S6","S7","S8","S9","S11","S13","S14","S15","S16","S17"]

    """We need to define the dataset path and the subject ID"""

    data_set_path = "/Users/kakis/OneDrive/Documenten/unie/Delft/BSc Electrical Engineering/Year 2/EPO/EPO-4/WESAD/WESAD/"
    def __init__(self,s):
        
        self.s=s
        obj_data = {}
        
        # Accessing class attributes and method through objects
        obj_data[s] = read_data_of_one_subject(self.data_set_path, s)

        
        """The code below allows you to read and print the length of all biosignals from the chest recording device recorded all at 700 Hz."""

        chest_data_dict = obj_data[self.s].get_chest_data()# your code here
        chest_dict_length = {key: len(value) for key, value in chest_data_dict.items()}
        
        """You can also access the labels of each class. The following labels are provided: 0 = not defined / transient, 1 = baseline, 2 = stress, 3 = amusement, 4 = meditation, 5/6/7 = should be ignored in this dataset."""
        
        # Get labels
        labels = obj_data[self.s].get_labels()
        baseline = np.asarray([idx for idx,val in enumerate(labels) if val == 1])
        stress = np.asarray([idx for idx,val in enumerate(labels) if val == 2]) 
        
        """Let's load some part of the ECG signal during baseline recording"""
        self.eda_base=chest_data_dict['EDA'][baseline,0]
        self.eda_stress=chest_data_dict['EDA'][stress,0]
        self.emg_base=chest_data_dict['EMG'][baseline,0]
        self.emg_stress=chest_data_dict['EMG'][stress,0]
        self.ecg_base=chest_data_dict['ECG'][baseline,0]
        self.ecg_stress=chest_data_dict['ECG'][stress,0]
        self.acc_base=chest_data_dict['ACC'][baseline,0]
        self.acc_stress=chest_data_dict['ACC'][stress,0]
        self.temp_base=chest_data_dict['Temp'][baseline,0]
        self.temp_stress=chest_data_dict['Temp'][stress,0]
        self.resp_stress = chest_data_dict['Resp'][stress, 0]
        self.resp_base = chest_data_dict['Resp'][baseline, 0]
        
        """You can also access the labels of each class. The following labels are provided: 0 = not defined / transient, 1 = baseline, 2 = stress, 3 = amusement, 4 = meditation, 5/6/7 = should be ignored in this dataset."""
        
        self.timestamp=np.arange(0,self.ecg_base.size*(1/self.Fs),(1/self.Fs))
        self.timestamp=self.timestamp[:self.ecg_base.size]
        self.t_stress=np.arange(0,self.ecg_stress.size*(1/self.Fs),(1/self.Fs))
        self.t_stress=self.t_stress[:self.ecg_stress.size]
        
        # Get labels
        # labels = obj_data[self.s].get_labels()
        # baseline = np.asarray([idx for idx,val in enumerate(labels) if val == 1])
        # stress = np.asarray([idx for idx,val in enumerate(labels) if val == 2]) 
        
        # wrist_data_dict = obj_data[self.s].get_wrist_data()# your code here
        # wrist_dict_length = {key: len(value) for key, value in wrist_data_dict.items()}
        
        # """Let's load some part of the ECG signal during baseline recording"""
        # self.eda_base_w=wrist_data_dict['EDA'][baseline,0]
        # self.eda_stress_w=wrist_data_dict['EDA'][stress,0]
        # self.emg_base_w=wrist_data_dict['EMG'][baseline,0]
        # self.emg_stress_w=wrist_data_dict['EMG'][stress,0]
        # self.ecg_base_w=wrist_data_dict['ECG'][baseline,0]
        # self.ecg_stress_w=wrist_data_dict['ECG'][stress,0]
        # self.acc_base_w=wrist_data_dict['ACC'][baseline,0]
        # self.acc_stress_w=wrist_data_dict['ACC'][stress,0]
        # self.temp_base_w=wrist_data_dict['Temp'][baseline,0]
        # self.temp_stress_w=wrist_data_dict['Temp'][stress,0]
        
        # def get_wrist_data(self):
        #     """"""
        #     #label = self.data[self.keys[0]]
        #     # assert subject == self.data[self.keys[1]]
        #     signal = self.data[self.keys[2]]
        #     wrist_data = signal[self.signal_keys[0]]
        #     #wrist_ACC = wrist_data[self.wrist_sensor_keys[0]]
        #     #wrist_ECG = wrist_data[self.wrist_sensor_keys[1]]
        #     return wrist_data