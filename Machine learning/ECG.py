# import os
import numpy as np
import neurokit2 as nk
from scipy.signal import butter, iirnotch, lfilter
import matplotlib.pyplot as plt
import sys
from scipy.signal import freqz
sys.path.append("/usr/local/lib/python3.7/site-packages")


# # =================loading data================================================
# # 
# """We need to define the dataset path and the subject ID"""
# #
# data_set_path = "/Users/kakis/OneDrive/Documenten/unie/Delft/BSc Electrical Engineering/Year 2/EPO/EPO-4/WESAD/WESAD/"
# subject = 'S2'# --- 'S17'
# #
# """The class below allows you to access all the recordings and labels in the dataset."""
# #
# #
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
# #    
# # Object instantiation
# obj_data = {}
# # 
# # Accessing class attributes and method through objects
# obj_data[subject] = read_data_of_one_subject(data_set_path, subject)
# #
# """The code below allows you to read and print the length of all biosignals from the chest recording device recorded all at 700 Hz."""
# #
# chest_data_dict = obj_data[subject].get_chest_data()# your code here
# chest_dict_length = {key: len(value) for key, value in chest_data_dict.items()}
# print(chest_dict_length)
# #
# """You can also access the labels of each class. The following labels are provided: 0 = not defined / transient, 1 = baseline, 2 = stress, 3 = amusement, 4 = meditation, 5/6/7 = should be ignored in this dataset."""
# #
# # Get labels
# labels = obj_data[subject].get_labels()
# baseline = np.asarray([idx for idx,val in enumerate(labels) if val == 1])
# stress = np.asarray([idx for idx,val in enumerate(labels) if val == 2])
# amusement = np.asarray([idx for idx,val in enumerate(labels) if val == 3])
# meditation = np.asarray([idx for idx,val in enumerate(labels) if val == 4])
# #
# #
# """Let's load some part of the ECG signal during baseline recording"""
# #
# fs=700
# ecg_base=chest_data_dict['ECG'][baseline,0]
# ecg_stress=chest_data_dict['ECG'][stress,0]
# ecg_amusement=chest_data_dict['ECG'][amusement,0]
# ecg_meditation=chest_data_dict['ECG'][meditation,0]
# # 
# # =============================================================================
# ECG = {}

# #read cvs file
# data = pd.read_csv("EMG_and_ECG.csv")
# data
# print(data.shape)
# print(data)

# # seperate file into EMG and ECG
# timestamp = data['TimeStamp']
# EMG = data['EMG Data']
# ECG = data['ECG Data']
# ecg_base = ECG
# #plot ECG
# plt.figure(figsize=(12,4))
# plt.plot(timestamp,ECG, label="raw ECG")
# plt.xlabel('$Time (s)$') 
# plt.ylabel('$ECG$') 
# plt.legend()
# plt.show()

# fs=200
class ECGprep:
    def __init__ (self,Fs,ecg_class=[],title=""): 
        self.title = title
        self.Fs=Fs
        self.ecg = ecg_class[10000:10000+ 10*self.Fs]
        self.t=np.arange(0,self.ecg.size*(1/self.Fs),(1/self.Fs))
        self.t=self.t[:self.ecg.size]
    def plotdata(self,ecg_class=1,title=""):
        if ecg_class==1:
            ecg=self.ecg
            t=self.t
        else:
            ecg = ecg_class[10000:10000+ 10*self.Fs]
            t=np.arange(0,self.emg.size*(1/self.Fs),(1/self.Fs))
            t=t[:self.ecg.size]
        # cut a smaller window      
        if title=="":
            title=self.title
        plt.figure(figsize=(12,4))
        plt.plot(t,ecg)
        plt.title("input ECG "+ title)
        plt.xlabel('$Time (s)$') 
        plt.xlim(0,max(t))
        plt.ylabel('$ECG$')
        plt.show()  
    def filtering_data(self,ecg_class=1,title=""):
        nyq = 0.5*self.Fs
        order=5
        
        if ecg_class==1:
            ecg=self.ecg
            t=self.t
        else:
            ecg = ecg_class[10000:10000+ 10*self.Fs]
            t=np.arange(0,self.emg.size*(1/self.Fs),(1/self.Fs))
            t=t[:self.ecg.size]

        # highpass filter
        high=0.5
        high= high/nyq
        b, a = butter(order, high, btype = 'highpass')
        ecg_h = lfilter(b,a,self.ecg)
        
        w, h = freqz(b, a, fs=self.Fs)
        
        fig, ax = plt.subplots(1, 2, figsize=(15,8))
        
        ax[0].plot(w/2/np.pi*self.Fs,abs(h))

        # lowpass filter
        low=70
        low= low/nyq
        b, a = butter(order, low)
        ecg_hl = lfilter(b,a,ecg_h)
        
        w, h = freqz(b, a, fs=self.Fs)
        
        ax[0].plot(w/2/np.pi*self.Fs,abs(h))
        ax[0].set_title("Butterworth bandpass filter frequency domain")
        ax[0].set_xlabel("Frequency [Hz]")
        ax[0].set_ylabel("ECG")
        ax[0].set_xscale("log")
        ax[0].margins(0,0.1)
        ax[0].grid(which="both", axis="both")
        ax[0].axvline(high*nyq, color="green")
        

        # notch filter remove powerline interference
        notch=50
        notch = notch/nyq
        b, a = iirnotch(notch,30,fs=self.Fs)
        ecg_hln = lfilter(b,a,ecg_hl)
        
        w, h = freqz(b, a, fs=self.Fs)
        
        ax[1].plot(w/2/np.pi*self.Fs,abs(h))
        ax[1].set_title("Butterworth bandpass and notch filter frequency domain")
        ax[1].set_xlabel("Frequency [Hz]")
        ax[1].set_ylabel("$ECG$")
        ax[1].set_xscale("log")
        ax[1].margins(0,0.1)
        ax[1].grid(which="both", axis="both")
        ax[1].axvline(high*nyq, color="green")
        plt.show()
        

        plt.figure(figsize=(12,4))
        plt.plot(t,ecg,label="raw ECG")
        plt.plot(t,ecg_hln, label="filtered ECG")
        if title=="":
            title=self.title
        plt.title("ECG filtering "+ title)
        plt.xlabel('$Time (s)$') 
        plt.xlim(0,max(t))
        plt.ylabel('$ECG$') 
        plt.legend()
        plt.show()

        plt.figure(figsize=(12,4))
        plt.plot(t,ecg_hln)
        plt.title("ECG filtered "+title)
        plt.xlabel('$Time (s)$') 
        plt.xlim(0,max(t))
        plt.ylabel('$ECG$') 
        plt.show()
        return ecg_hln

# ECG = ECGprep(fs,ecg_base,"baseline")
   
# ECG.plotdata()
# ecg_filt = ECG.filtering_data()

# ECG = ECGprep(fs,ecg_stress,"stress")
   
# ECG.plotdata()
# ecg_filt = ECG.filtering_data()

# ECG = ECGprep(fs,ecg_amusement,"amusement")
   
# ECG.plotdata()
# ecg_filt = ECG.filtering_data()

# ECG = ECGprep(fs,ecg_meditation,"meditation")
   
# ECG.plotdata()
# ecg_filt = ECG.filtering_data()

ECGfeat = {}
class ECGfeatures:
    def __init__(self,ecg_filt,Fs,title=""):
        self.ecg = ecg_filt
        self.Fs = Fs
        self.title=title
        
    def rpeaks(self,title=""):
        _, rpeaks = nk.ecg_peaks(self.ecg, sampling_rate=self.Fs) 
        if title=="":
            title=self.title
        plt.plot((rpeaks['ECG_R_Peaks']/self.Fs),self.ecg[rpeaks['ECG_R_Peaks']], 'go')
        t=np.arange(0,self.ecg.size*(1/self.Fs),(1/self.Fs))
        plt.plot(t,self.ecg)
        plt.title("R-peaks "+ title)
        plt.xlim(0,max(t))
        plt.xlabel('$Time (s)$') 
        plt.ylabel('$ECG$') 
        plt.show()
        
# ECGfeat = ECGfeatures(ecg_filt,fs,"test")
# ECGfeat.rpeaks("baseline")
# ECGfeat.rpeaks()

        