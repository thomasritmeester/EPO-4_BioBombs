from scipy.signal import butter, sosfiltfilt
import matplotlib.pyplot as plt
import numpy as np
import pickle
from DATABASES import plotting as An

class TEMPprep:
    Fs=700
    if An.wesad == False:
        g=1000000
    else:
        g= Fs
    def __init__ (self,timestamp,title=""): 
        self.title = title
        self.timestamp = timestamp
        self.plot=False
    def alles(self, sig, plot=False):
        self.plot=plot
        if plot == True:
            self.plotdata(sig)
        self.Result = self.filtering_data(sig)
        return self.tempH
    
    def plotdata(self,sig,title=""):
        # cut a smaller window      
        if title=="":
            title=self.title
        plt.figure(figsize=(12,4))
        plt.plot(self.timestamp/self.g,sig)
        plt.title("raw TEMP ("+ title+")")
        plt.xlabel('$Time (s)$') 
        plt.xlim(0,max(self.timestamp)/self.g)
        plt.ylabel('$Magnitude$')
        plt.show()
        
    def filtering_data(self,sig):
        
        nyq=self.Fs/2

        order=5
        freqs=200/nyq
        sos = butter(order, freqs, btype='low', output="sos",  fs= self.Fs)
        tempH= sosfiltfilt(sos, sig)
        
        if self.plot == True:
            plt.figure(figsize=(12,4))
            plt.plot(self.timestamp/self.g,sig, label='raw Temp')
            plt.plot(self.timestamp/self.g,tempH, label="Lowpass filtered Temp")
            plt.title("Filtering temperature ("+self.title+")")
            plt.xlabel('$Time (s)$') 
            plt.ylabel('$Magnitude$')
            plt.legend()
            plt.show()   
            
        self.tempH= tempH
        return tempH
    
    
# """The class below allows you to access all the recordings and labels in the dataset."""


# class read_data_of_one_subject:
#     """Read data from WESAD dataset"""
#     def __init__(self, path, subject):
#         self.keys = ['label', 'subject', 'signal']
#         self.signal_keys = ['wrist', 'chest']
#         self.chest_sensor_keys = ['ACC', 'ECG', 'EDA', 'EMG', 'Resp', 'Temp']
#         self.wrist_sensor_keys = ['ACC', 'BVP', 'EDA', 'TEMP']
#         #os.chdir(path)
#         #os.chdir(subject)
#         with open(path + subject +'/'+subject + '.pkl', 'rb') as file:
#             data = pickle.load(file, encoding='latin1')
#         self.data = data


#     def get_labels(self):
#         return self.data[self.keys[0]]
    

#     def get_wrist_data(self):
#         """"""
#         #label = self.data[self.keys[0]]
#         assert subject == self.data[self.keys[1]]
#         signal = self.data[self.keys[2]]
#         wrist_data = signal[self.signal_keys[0]]
#         #wrist_ACC = wrist_data[self.wrist_sensor_keys[0]]
#         #wrist_ECG = wrist_data[self.wrist_sensor_keys[1]]
#         return wrist_data

#     def get_chest_data(self):
#         """"""
#         signal = self.data[self.keys[2]]
#         chest_data = signal[self.signal_keys[1]]
#         return chest_data
    

# Object instantiation
# obj_data = {}
# TEMP ={}

# data_set_path = "/Users/kakis/OneDrive/Documenten/unie/Delft/BSc Electrical Engineering/Year 2/EPO/EPO-4/WESAD/WESAD/"
# subject = ["S2","S3","S4","S5","S6","S7","S8","S9","S10","S11","S13","S14","S15","S16","S17"]
 
# for s in subject: 
#     # Accessing class attributes and method through objects
#     obj_data[s] = read_data_of_one_subject(data_set_path, s)
    
#     """The code below allows you to read and print the length of all biosignals from the chest recording device recorded all at 700 Hz."""
    
#     chest_data_dict = obj_data[s].get_chest_data()
#     chest_dict_length = {key: len(value) for key, value in chest_data_dict.items()}
#     print(s)
    
#     """You can also access the labels of each class. The following labels are provided: 0 = not defined / transient, 1 = baseline, 2 = stress, 3 = amusement, 4 = meditation, 5/6/7 = should be ignored in this dataset."""
    
#     # Get labels
#     labels = obj_data[s].get_labels()
#     baseline = np.asarray([idx for idx,val in enumerate(labels) if val == 1])
#     stress = np.asarray([idx for idx,val in enumerate(labels) if val == 2])

    
#     """Let's load some part of the TEMP signal during stress recording"""
#     temp_stress=chest_data_dict['Temp'][stress,0]
#     # cut a smaller window
#     fs=200
#     temp=temp_stress
#     t=np.arange(0,temp.size*(1/fs),(1/fs))[:temp.size]
#     # t=t[:temp.size]
      
#     TEMP = TEMPprep(t,fs)
#     TEMP_filt = TEMP.filtering_data(temp)


