# import os
import numpy as np
import neurokit2 as nk
from scipy.signal import butter, iirnotch, lfilter
import matplotlib.pyplot as plt
from scipy.signal import freqz
from DATABASES import plotting as An

class ECGprep:
    Fs=700
    if An.wesad == False:
        g=1000000
    else:
        g= Fs
    def __init__ (self,ecg_class,title=""): 
        self.title = title
        self.ecg = ecg_class
        self.plot=False
        self.t=np.arange(0,self.ecg.size*(1/self.g),(1/self.g))
        self.t=self.t[:self.ecg.size]
        
    def alles(self, plot=False):
        self.plot=plot
        if plot == True:
            self.plotdata()
        self.Result = self.filtering_data()
        return self.ecg_filt
        
    def plotdata(self,ecg_class=1,title=""):
        if ecg_class==1:
            ecg=self.ecg
            t=self.t
        else:
            ecg = ecg_class
            t=np.arange(0,self.ecg.size*(1/self.g),(1/self.g))
            t=t[:self.ecg.size]
        # cut a smaller window      
        if title=="":
            title=self.title
        plt.figure(figsize=(12,4))
        plt.plot(t,ecg)
        plt.title("input ECG ("+ title+")")
        plt.xlabel('$Time (s)$') 
        plt.xlim(0,max(t))
        plt.ylabel('$Magnitude$')
        plt.show()  
        
    def filtering_data(self,ecg_class=1,title=""):
        nyq = 0.5*self.Fs
        order=5
        
        if ecg_class==1:
            ecg=self.ecg
            t=self.t
        else:
            ecg = ecg_class
            t=np.arange(0,self.ecg.size*(1/self.g),(1/self.g))
            t=t[:self.ecg.size]

        # highpass filter
        high=0.5
        high= high/nyq
        b, a = butter(order, high, btype = 'highpass')
        ecg_h = lfilter(b,a,ecg)
        
        if self.plot == True:
        
            w, h = freqz(b, a, fs=self.Fs)
            
            plt.figure()
            plt.plot(w/2/np.pi*self.Fs,abs(h))

        # lowpass filter
        low=70
        low= low/nyq
        b, a = butter(order, low)
        ecg_hl = lfilter(b,a,ecg_h)
        
        if self.plot == True:
        
            w, h = freqz(b, a, fs=self.Fs)
            
            plt.plot(w/2/np.pi*self.Fs,abs(h))
            plt.title("Butterworth bandpass filter")
            plt.xlabel("Frequency [Hz]")
            plt.ylabel("Magnitude")
            plt.xscale("log")
            plt.margins(0,0.1)
            plt.grid(which="both", axis="both")
            plt.axvline(high*nyq, color="green")
            plt.show()

        # notch filter remove powerline interference
        notch=50
        notch = notch/nyq
        b, a = iirnotch(notch,30,fs=self.Fs)
        ecg_hln = lfilter(b,a,ecg_hl)
        self.ecg_filt = ecg_hln
        
        if self.plot == True:
        
            w, h = freqz(b, a, fs=self.Fs)
            
            plt.figure()
            plt.plot(w/2/np.pi*self.Fs,abs(h))
            plt.title("Notch filter ")
            plt.xlabel("Frequency [Hz]")
            plt.ylabel("$Magnitude$")
            plt.xscale("log")
            plt.margins(0,0.1)
            plt.grid(which="both", axis="both")
            plt.axvline(high*nyq, color="green")
            plt.show()
            
    
            plt.figure(figsize=(12,4))
            plt.plot(t,ecg,label="raw ECG")
            plt.plot(t[::7],ecg[::7], label="downsampled ECG")
            if title=="":
                title=self.title
            plt.title("ECG filtering ("+ title+")")
            plt.xlabel('$Time (s)$') 
            plt.xlim(0,max(t))
            plt.ylabel('$Magnitude$') 
            plt.legend()
            plt.show()
    
            # plt.figure(figsize=(12,4))
            # plt.plot(t,ecg_hln)
            # plt.title("ECG filtered ("+ title+")")
            # plt.xlabel('$Time (s)$') 
            # plt.xlim(0,max(t))
            # plt.ylabel('$ECG$') 
            # plt.show()
        return ecg_hln
   

class ECGfeatures:
    
    def __init__(self,ecg_filt=[],Fs=700,title=""):
        self.ecg = ecg_filt
        self.Fs = Fs
        self.title= title
        ECGprep.filtering_data(ecg_filt)
        
        
    def rpeaks(self,title=""):
        _, rpeaks = nk.ecg_peaks(self.ecg, sampling_rate=self.Fs) 
        # if title=="":
        #     title=self.title
        # plt.plot((rpeaks['ECG_R_Peaks']/self.Fs),self.ecg[rpeaks['ECG_R_Peaks']], 'go')
        # t=np.arange(0,self.ecg.size*(1/self.Fs),(1/self.Fs))
        # plt.plot(t,self.ecg)
        # plt.title("R-peaks "+ title)
        # plt.xlim(0,max(t))
        # plt.xlabel('$Time (s)$') 
        # plt.ylabel('$ECG$') 
        # plt.show()
        return rpeaks
        
# ECGfeat = ECGfeatures(ecg_filt,fs,"test")
# ECGfeat.rpeaks("baseline")
# ECGfeat.rpeaks()

        