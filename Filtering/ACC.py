import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt
from DATABASES import plotting as An

class ACCprep:
    Fs=200
    if An.wesad == False:
        g=1000000
    else:
        g= Fs
    def __init__ (self,timestamp,title=""): 
        self.title = title
        self.timestamp = timestamp
        self.plot=False
        
    def alles(self, sig, plot=False):
        if plot == True:
            self.plot=plot
            self.plotdata(sig)
        self.Result = self.filtering_data(sig)
        return self.sig_HL
        
    def plotdata(self,sig,title=""):
        # cut a smaller window      
        if title=="":
            title=self.title
        plt.figure(figsize=(12,4))
        plt.plot(self.timestamp/self.g,sig)
        plt.title("raw ACC ("+ title+")")
        plt.xlabel('$Time (s)$') 
        plt.xlim(0,max(self.timestamp)/self.g)
        plt.ylabel('$Magnitude$')
        plt.show()
        
    def filtering_data(self,sig):
        nyq=self.Fs/2
        corner=0.7
        corner= corner/nyq
        order = 2
        sos = butter(order, corner, btype = 'highpass', output='sos')

        sig_H= sosfiltfilt(sos,sig)###################
        #w, h = freqz(b, a, fs=Fs)
        
        order = 2
        sos = butter(order, corner, btype = 'lowpass', output='sos')
        sig_HL= sosfiltfilt(sos,sig_H)
        self.sig_HL = sig_HL
        
        if self.plot == True:   
            plt.figure(figsize=(10,4))
            plt.plot(self.timestamp/self.g,sig - np.mean(sig), label="raw ACC") ###########################
            plt.plot(self.timestamp/self.g,sig_HL*10, label="ACC bandpass filtered")
            plt.title("Filtered ACC signal ("+self.title+")")
            plt.xlabel('$Time (s)$') 
            plt.ylabel('$Magnitude$') 
            plt.legend()
            plt.show()
        return sig_HL

