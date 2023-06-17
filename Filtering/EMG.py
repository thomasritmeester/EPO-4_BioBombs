import numpy as np
from scipy.fft import fft
from scipy.signal import butter, iirnotch, lfilter, sosfilt
from scipy import signal as sig
import matplotlib.pyplot as plt
from DATABASES import plotting as An

EMG = {}
class EMGprep:
    Fs=700
    if An.wesad == False:
        g=1000000
    else:
        g= Fs
    def __init__ (self,emg_class=[],title=""): 
        self.title = title
        self.emg = emg_class
        self.plot=False
        self.t=np.arange(0,self.emg.size*(1/self.g),(1/self.g))
        self.t=self.t[:self.emg.size]
        
    def alles(self, plot=False):
        self.plot=plot
        if plot == True:
            self.plotdata()
        self.Result = self.filtering_data()
        return self.emg_bn
        
    def plotdata(self,emg_class=[],title=""):
        if len(emg_class) == 0:
            emg=self.emg
            t=self.t
        else:
            emg = emg_class
            t=np.arange(0,self.emg.size*(1/self.g),(1/self.g))
            t=t[:self.emg.size]
            if len(self.emg) == 0:
                self.emg = emg
        # cut a smaller window      
        if title=="":
            title=self.title
        plt.figure(figsize=(7,4))
        plt.plot(t,emg)
        plt.title("input EMG ("+title+")")
        plt.xlim(0,max(t))
        plt.xlabel('$Time (s)$') 
        plt.ylabel('$Magnitude$')
        plt.show()  
        
        
        X = fft(emg)
        
        freqs = np.arange(0, self.Fs, self.Fs/len(X))
        plt.figure(figsize=(7,4))
        plt.plot(freqs,(abs(X)))
        plt.title("Input in frequency domain ("+title+")")
        plt.xlabel("Frequency [Hz]")
        plt.xlim(0-self.Fs/len(X),self.Fs/2)
        plt.ylim(-max(abs(X)[200:])*0.05,max(abs(X)[200:])*1.2)
        plt.ylabel("$Magnitude$")
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
            emg = emg_class
            t=np.arange(0,self.emg.size*(1/self.g),(1/self.g))
            t=t[:self.emg.size]
            if len(self.emg) == 0:
                self.emg = emg
            
        if title=="":
            title=self.title

        # bandpass filter
        high = 50/nyq
        sos1 = butter(order, high, btype ="highpass", output='sos') #sos more stable the b,a
        emg_filt = sosfilt(sos1, self.emg)
        w1,h1=sig.sosfreqz(sos1)
        
        low= 300/nyq 
        sos = butter(order, low, btype="lowpass", output="sos")
        emg_filt = sosfilt(sos, emg_filt)  
        emg_b = emg_filt
        
        if self.plot == True:
           
            plt.figure(figsize=(6.5,4))
            plt.plot(t,emg,label="raw EMG")
            plt.plot(t,emg_b, label="bandpass filtered EMG")
            plt.title("EMG bandpass filtered ("+ title+")")
            plt.xlim(0,max(t))
            plt.xlabel('$Time (s)$') 
            plt.ylabel('$Magnitude$') 
            plt.legend()
            plt.show()
            
           
            w,h=sig.sosfreqz(sos)
            plt.figure(figsize=(6.5,4))
            plt.plot(w1/2/np.pi*self.Fs,abs(h1))
            plt.plot(w/2/np.pi*self.Fs,abs(h))
            plt.title("Butterworth bandpass filter")
            plt.xlabel("Frequency [Hz]")
            plt.ylabel("$Magnitude$")
            plt.margins(0,0.1)
            plt.grid(which="both", axis="both")
            plt.axvline(high*nyq, color="green",ls ='--')
            plt.axvline(low*nyq, color="green",ls ='--')
            plt.show() 
            
            X = fft(emg_b)
            
            freqs = np.arange(0, self.Fs, self.Fs/len(X))
            
            plt.figure(figsize=(6.5,4))
            plt.plot(freqs,(abs(X)-min(abs(X)))/(max(abs(X))-min(abs(X))))
            plt.xlim(0-self.Fs/len(X),self.Fs/2)
            plt.title("EMG signal bandpass filtered in frequency domain ("+title+")")
            plt.xlabel("Frequency [Hz]")
            plt.ylabel("$Magnitude$")
            plt.grid(which="both", axis="both")
            plt.show()

        # notch filter remove powerline interference
        notchs=[0,1,2,3,4,5,6,7]*50
        for notch in notchs:
            notch = notchs[notch]/nyq
            b, a = iirnotch(notch,30,fs=self.Fs)
            emg_b = lfilter(b,a,emg_b)
        emg_bn = emg_b
        self.emg_bn = emg_bn
        
        if self.plot == True:
        
            X = fft(emg_bn)
            
            freqs = np.arange(0, self.Fs, self.Fs/len(X))
            
            plt.figure(figsize=(6.5,4))
            plt.plot(freqs,abs(X))
            plt.xlim(0-self.Fs/len(X),self.Fs/2)
            plt.title("EMG signal bandpass and Notch filtered in frequency domain ("+title+")")
            plt.xlabel("Frequency [Hz]")
            plt.ylabel("$Magnitude$")
            plt.grid(which="both", axis="both")
            plt.show()
            
            plt.figure(figsize=(10,3.7))
            plt.plot(t,emg,label="raw EMG lowered by the mean of signal")
            plt.plot(t,emg_bn, label="filtered EMG")
            plt.title("EMG filter output ("+ title+")")
            plt.xlim(0,max(t))
            plt.xlabel('$Time (s)$') 
            plt.ylabel("Magnitude")
            # plt.ylim(min(emg_bn)-0.02,max(emg_bn)+0.02)
            plt.legend()
            plt.show()  
        
        return emg_bn
    

    

