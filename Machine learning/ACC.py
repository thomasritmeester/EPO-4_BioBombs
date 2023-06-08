import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt


# =============================================================================
# #read cvs file
# data = pd.read_csv("completetest_stress.csv")
# #print(data)
# Fs = 200 
# timestamp = data['TimeStamp'].to_numpy()
# ACC1x = data['Acc1 X'].to_numpy()
# ACC1y = data['Acc1 Y'].to_numpy()
# ACC1z = data['Acc1 z'].to_numpy()
# ACC2x = data['Acc2 X'].to_numpy()
# ACC2y = data['Acc2 Y'].to_numpy()
# ACC2z = data['Acc2 z'].to_numpy()
# =============================================================================

class ACCprep:
    def __init__ (self,Fs,timestamp,title=""): 
        self.title = title
        self.Fs=Fs
        self.timestamp = timestamp
    # def plotdata(self,sig,title=""):
    #     # cut a smaller window      
    #     # if title=="":
    #     #     title=self.title
    #     # plt.figure(figsize=(12,4))
    #     # plt.plot(self.timestamp,sig)
    #     # plt.title("input ACC "+ title)
    #     # plt.xlabel('$Time (s)$') 
    #     # plt.xlim(0,max(self.timestamp))
    #     # plt.ylabel('$ACC$')
    #     # plt.show()
        
    def filtering_data(self,sig):
        nyq=self.Fs/2
        corner=0.5
        corner= corner/nyq
        order = 2
        sos = butter(order, corner, btype = 'highpass', output='sos')
        print(sig)
        sig_H= sosfiltfilt(sos,sig)###################
        #w, h = freqz(b, a, fs=Fs)
        
        order = 2
        sos = butter(order, corner, btype = 'lowpass', output='sos')
        sig_HL= sosfiltfilt(sos,sig_H)
        
        # plt.figure(figsize=(10,4))
        # plt.plot(self.timestamp/self.Fs,sig - np.mean(sig), label="raw ACC") ###########################
        # plt.plot(self.timestamp/self.Fs,sig_HL*10, label="ACC bandpass filtered")
        # plt.title("Filtered ACC signal")
        # plt.xlabel('$Time (s)$') 
        # plt.ylabel('$ACC$') 
        # plt.legend()
        # plt.show()
        return sig_HL
        
# ACC ={}
# ACC = ACCprep(Fs,timestamp)
# print(ACC1x)
# ACC.plotdata(ACC1x)
# ACC1x_filt = ACC.filtering_data(ACC1x)
