# import os
# from os import listdir
# from os.path import isfile, join, isdir
import pickle
import numpy as np
from scipy.fft import fft
import pandas as pd
from scipy.signal import butter, iirnotch, lfilter, sosfilt
from scipy import signal as sig
# import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy.ndimage


EMG = {}
class EMGprep:
    def __init__ (self,Fs,emg_class=[],title=""): 
        self.title = title
        self.Fs=Fs
        self.emg = emg_class
        self.t=np.arange(0,self.emg.size*(1/self.Fs),(1/self.Fs))
        self.t=self.t[:self.emg.size]
    def plotdata(self,emg_class=[],title=""):
        if len(emg_class) == 0:
            emg=self.emg
            t=self.t
        else:
            emg = emg_class[:]
            t=np.arange(0,self.emg.size*(1/self.Fs),(1/self.Fs))
            t=t[:self.emg.size]
        # cut a smaller window      
        if title=="":
            title=self.title
        plt.figure()
        plt.plot(t,emg)
        plt.title("input EMG "+ title)
        plt.xlim(0,max(t))
        plt.xlabel('$Time (s)$') 
        plt.ylabel('$EMG$')
        plt.show()  
        
        
        X = fft(emg)
        
        freqs = np.arange(0, self.Fs, self.Fs/len(X))
        plt.figure()
        plt.plot(freqs,20*np.log10(abs(X)))
        plt.title("Input in frequency domain ("+title+")")
        plt.xlabel("Frequency [Hz]")
        plt.xlim(0-self.Fs/len(X),self.Fs/2)
        plt.ylabel("$EMG$")
        plt.margins(0,0.1)
        plt.grid(which="both", axis="both")
        plt.show()
        
        
    def filtering_data(self,emg_class=[],title=""):
        nyq = 0.5*self.Fs
        order=4
          
        if len(emg_class) == 0:
            emg=self.emg
            t=self.t
        else:
            emg = emg_class[10000:10000+ 10*self.Fs]
            t=np.arange(0,self.emg.size*(1/self.Fs),(1/self.Fs))
            t=t[:self.emg.size]
            
        if title=="":
            title=self.title

        # highpass filter
        high = 50/nyq
        sos = butter(order, high, btype ="highpass", output='sos') #sos more stable the b,a
        emg_h = sosfilt(sos, self.emg)
#         fig, ax = plt.subplots(2, 2, figsize=(15,8))
        
#         ax[0,0].plot(t,emg,label="raw EMG")
#         ax[0,0].plot(t,emg_h, label="highpass filtered EMG")
#         ax[0,0].set_title("EMG highpass filtered ("+ title+")")
#         ax[0,0].set_xlim(0,max(t))
#         ax[0,0].set_xlabel('$Time (s)$') 
#         ax[0,0].set_ylabel('$EMG$') 
#         ax[0,0].legend()
        # ax[0,0].show()
        
       
        w,h=sig.sosfreqz(sos)
#         ax[0,1].plot(w/2/np.pi*self.Fs,abs(h))
#         ax[0,1].set_title("Butterworth highpass filter")
#         ax[0,1].set_xlabel("Frequency [Hz]")
#         ax[0,1].set_ylabel("$EMG$")
#         ax[0,1].margins(0,0.1)
#         ax[0,1].grid(which="both", axis="both")
#         ax[0,1].axvline(high*nyq, color="green")
        # plt.show() 
        
        X = fft(emg_h)
        
#         freqs = np.arange(0, self.Fs, self.Fs/len(X))
        
#         ax[1,1].plot(freqs,abs(X))
#         ax[1,1].set_xlim(0-self.Fs/len(X),self.Fs/2)
#         ax[1,1].set_title("EMG signal highpass filtered in frequency domain ("+title+")")
#         ax[1,1].set_xlabel("Frequency [Hz]")
#         ax[1,1].set_ylabel("$EMG$")
#         ax[1,1].grid(which="both", axis="both")
        # plt.show()

        # notch filter remove powerline interference
        notchs=[0,1,2,3,4,5,6,7]*50
        for notch in notchs:
            notch = notchs[notch]/nyq
            b, a = iirnotch(notch,30,fs=self.Fs)
            emg_h = lfilter(b,a,emg_h)
        emg_hn = emg_h
        
        X = fft(emg_hn)
        
        freqs = np.arange(0, self.Fs, self.Fs/len(X))
        
#         ax[1,0].plot(freqs,abs(X))
#         ax[1,0].set_xlim(0-self.Fs/len(X),self.Fs/2)
#         ax[1,0].set_title("EMG signal highpass and Notch filtered in frequency domain ("+title+")")
#         ax[1,0].set_xlabel("Frequency [Hz]")
#         ax[1,0].set_ylabel("$EMG$")
#         ax[1,0].grid(which="both", axis="both")
#         plt.show()
        
#         plt.figure(figsize=(12,4))
#         plt.plot(t,emg-np.mean(emg),label="raw EMG lowered by the mean of signal")
#         plt.plot(t,emg_hn, label="highpasss & notch filtered EMG")
#         plt.title("EMG highpass filter output ("+ title+")")
#         plt.xlim(0,max(t))
#         plt.xlabel('$Time (s)$') 
#         plt.ylabel("V")
#         plt.ylim(min(emg_hn)-5,max(emg_hn)+5)
#         plt.legend()
#         plt.show()   

        # lowpass filter ##########################################
        low = 10/nyq
        sos = butter(order, low, btype ="lowpass", output='sos') #sos more stable the b,a
        emg_l = sosfilt(sos, self.emg)
        
        fig, ax = plt.subplots(2, 2, figsize=(15,8))
        
#         ax[0,0].plot(t,emg,label="raw EMG")
#         ax[0,0].plot(t,emg_l, label="lowpass filtered EMG ("+title+")")
#         ax[0,0].set_title("EMG lowpass filtering ("+ title+")")
#         ax[0,0].set_xlabel('$Time (s)$') 
#         ax[0,0].set_xlim(0,max(t))
#         ax[0,0].set_ylabel('$EMG$') 
#         ax[0,0].legend()
        # ax[0,0].show()
        
       
        w,h=sig.sosfreqz(sos)
#         ax[0,1].plot(w/2/np.pi*self.Fs,abs(h))
#         ax[0,1].set_xlim(0,self.Fs/2)
#         ax[0,1].set_title("Butterworth lowpass filter frequency domain ("+title+")")
#         ax[0,1].set_xlabel("Frequency [rad/s]")
#         ax[0,1].set_xlim(0,max(t))
#         ax[0,1].set_ylabel("$EMG$")
#         ax[0,1].margins(0,0.1)
#         ax[0,1].grid(which="both", axis="both")
#         ax[0,1].axvline(low*nyq, color="green")
        # plt.show() 
        
        X = fft(emg_l)
        
        freqs = np.arange(0, self.Fs, self.Fs/len(X))
        
#         ax[1,1].plot(freqs,abs(X))
#         ax[1,1].set_xlim(0-self.Fs/len(X),self.Fs/2)
#         ax[1,1].set_title("EMG signal lowpass filtered in frequency domain ("+title+")")
#         ax[1,1].set_xlabel("Frequency [Hz]")
#         ax[1,1].set_ylabel("$EMG$")
#         ax[1,1].grid(which="both", axis="both")
        # plt.show()

        # notch filter remove powerline interference
        notchs=[0,1,2,3,4,5,6,7]*50
        for notch in notchs:
            notch = notchs[notch]/nyq
            b, a = iirnotch(notch,30,fs=self.Fs)
            emg_l = lfilter(b,a,emg_l)
        emg_ln = emg_l
        
        X = fft(emg_ln)
        
        freqs = np.arange(0, self.Fs, self.Fs/len(X))
        
#         ax[1,0].plot(freqs,abs(X))
#         ax[1,0].set_xlim(0-self.Fs/len(X),self.Fs/2)
#         ax[1,0].set_title("EMG signal lowpass and Notch filtered in frequency domain ("+title+")")
#         ax[1,0].set_xlabel("Frequency [Hz]")
#         ax[1,0].grid(which="both", axis="both")
#         ax[1,0].set_ylabel("$EMG$")
#         plt.show()
        
#         plt.figure(figsize=(12,4))
#         plt.plot(t,emg- np.mean(emg),label="raw EMG lowered by the mean of signal")
#         plt.plot(t,emg_ln, label="lowpas & notch filtered EMG ("+title+")")
#         plt.title("EMG lowpass filter output ("+ title+")")
#         plt.xlabel('$Time (s)$')
#         plt.xlim(0,max(t))
#         plt.ylabel('$EMG$')
#         plt.ylim(min(emg_ln)-5,max(emg_ln)+5)
#         plt.legend()
#         plt.show()        
        
        return emg_hn
    
# EMG = EMGprep(fs,emg_base,"stress")
   
# EMG.plotdata()
# EMG.filtering_data()
# EMG.plotdata(emg_stress,"stress")
# EMG.filtering_data(emg_stress,"stress")
