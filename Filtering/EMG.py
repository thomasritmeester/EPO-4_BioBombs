import numpy as np
from scipy.fft import fft
from scipy.signal import butter, iirnotch, lfilter, sosfilt
from scipy import signal as sig
import matplotlib.pyplot as plt


EMG = {}
class EMGprep:
    def __init__ (self,Fs=650,emg_class=[],title=""): 
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

        # bandpass filter
        high = 50/nyq
        sos = butter(order, high, btype ="highpass", output='sos') #sos more stable the b,a
        emg_filt = sosfilt(sos, self.emg)
        w1,h1=sig.sosfreqz(sos)
        
        low= 300/nyq 
        sos = butter(order, low, btype="lowpass", output="sos")
        emg_filt = sosfilt(sos, emg_filt)  
        emg_b = emg_filt

        # notch filter remove powerline interference
        notchs=[0,1,2,3,4,5,6,7]*50
        for notch in notchs:
            notch = notchs[notch]/nyq
            b, a = iirnotch(notch,30,fs=self.Fs)
            emg_b = lfilter(b,a,emg_b)
        emg_bn = emg_b
        
        return emg_bn

