# import os
import numpy as np
import neurokit2 as nk
from scipy.signal import butter, iirnotch, lfilter
import matplotlib.pyplot as plt
import sys
from scipy.signal import freqz
from neurokit2.signal import signal_fixpeaks, signal_formatpeaks
sys.path.append("/usr/local/lib/python3.7/site-packages")

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
#         plt.figure(figsize=(12,4))
#         plt.plot(t,ecg)
#         plt.title("input ECG "+ title)
#         plt.xlabel('$Time (s)$') 
#         plt.xlim(0,max(t))
#         plt.ylabel('$ECG$')
#         plt.show()  
        
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
        
#         plt.figure()
#         plt.plot(w/2/np.pi*self.Fs,abs(h))

        # lowpass filter
        low=70
        low= low/nyq
        b, a = butter(order, low)
        ecg_hl = lfilter(b,a,ecg_h)
        
        w, h = freqz(b, a, fs=self.Fs)
        
#         plt.plot(w/2/np.pi*self.Fs,abs(h))
#         plt.title("Butterworth bandpass filter")
#         plt.xlabel("Frequency [Hz]")
#         plt.ylabel("ECG")
#         plt.xscale("log")
#         plt.margins(0,0.1)
#         plt.grid(which="both", axis="both")
#         plt.axvline(high*nyq, color="green")
#         plt.show()

        # notch filter remove powerline interference
        notch=50
        notch = notch/nyq
        b, a = iirnotch(notch,30,fs=self.Fs)
        ecg_hln = lfilter(b,a,ecg_hl)
        
        w, h = freqz(b, a, fs=self.Fs)
        
#         plt.figure()
#         plt.plot(w/2/np.pi*self.Fs,abs(h))
#         plt.title("Notch filter ")
#         plt.xlabel("Frequency [Hz]")
#         plt.ylabel("$ECG$")
#         plt.xscale("log")
#         plt.margins(0,0.1)
#         plt.grid(which="both", axis="both")
#         plt.axvline(high*nyq, color="green")
#         plt.show()
        

#         plt.figure(figsize=(12,4))
#         plt.plot(t,ecg,label="raw ECG")
#         plt.plot(t,ecg_hln, label="filtered ECG")
#         if title=="":
#             title=self.title
#         plt.title("ECG filtering "+ title)
#         plt.xlabel('$Time (s)$') 
#         plt.xlim(0,max(t))
#         plt.ylabel('$ECG$') 
#         plt.legend()
#         plt.show()

#         plt.figure(figsize=(12,4))
#         plt.plot(t,ecg_hln)
#         plt.title("ECG filtered "+title)
#         plt.xlabel('$Time (s)$') 
#         plt.xlim(0,max(t))
#         plt.ylabel('$ECG$') 
#         plt.show()
        return ecg_hln

# ECG = ECGprep(fs,ecg_base,"baseline")
   
# ECG.plotdata()
# ecg_filt = ECG.filtering_data()

# ECG = ECGprep(fs,ecg_stress,"stress")
   

class ECGfeatures:
    def __init__(self,ecg_filt,Fs,title=""):
        self.ecg = ecg_filt
        self.Fs = Fs
        self.title=title
        
    def rpeaks(self, correct_artifacts=False, title=""):
        rpeaks, info = nk.ecg_peaks(self.ecg, sampling_rate=self.Fs) 
        if title=="":
            title=self.title
        plt.plot((rpeaks['ECG_R_Peaks']/self.Fs),self.ecg[rpeaks['ECG_R_Peaks']], 'go')
        t=np.arange(0,self.ecg.size*(1/self.Fs),(1/self.Fs))
        plt.plot(t,self.ecg)
        plt.title("R-peaks "+ title)
        plt.xlim(0,max(t))
        plt.xlabel('$Time (s)$s') 
        plt.ylabel('$ECG$') 
        plt.show()

        if correct_artifacts:
            _, rpeaks = signal_fixpeaks(
                rpeaks, sampling_rate=self.Fs, iterative=True, method="Kubios"
            )

            rpeaks = {"ECG_R_Peaks": rpeaks}

        instant_peaks = signal_formatpeaks(rpeaks, desired_length=len(self.ecg), peak_indices=rpeaks)
        signals = instant_peaks
        info = rpeaks
        info["sampling_rate"] = self.Fs  # Add sampling rate in dict info
        return rpeaks, info
        
    def HRV(self, signal,title=""):
        #if signal==[]:
        #    signal=self.ecg
        #    print("yes")
        #else:
        #    print("No")
        HRV=nk.hrv(self, signal, sampling_rate=self.Fs, show=True)
        # if title=="":
        #     title=self.title
        #plt.plot((rpeaks['ECG_R_Peaks']/self.Fs),self.ecg[rpeaks['ECG_R_Peaks']], 'go')
        # t=np.arange(0,self.ecg.size*(1/self.Fs),(1/self.Fs))
        # plt.plot(t,self.ecg)
        # plt.title("R-peaks "+ title)
        # plt.xlim(0,max(t))
        # plt.xlabel('$Time (s)$') 
        # plt.ylabel('$ECG$') 
        # plt.show()
        return HRV
        
# ECGfeat = ECGfeatures(ecg_filt,fs,"test")
# ECGfeat.rpeaks("baseline")
# ECGfeat.rpeaks()

        
