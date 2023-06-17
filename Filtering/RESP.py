import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import numpy as np
import sys


class RESPprep:
    def __init__ (self,Fs,resp_class,title): 
        self.title = title
        self.Fs=Fs
        self.resp = resp_class #[10000:10000+ 10*self.Fs]
        self.t=np.arange(0,self.resp.size*(1/self.Fs),(1/self.Fs))
        self.t=self.t[:self.resp.size]
        
    def alles(self, plot=False):
        if plot == True:
            plt.figure()
            plt.plot(self.resp[:700*60])
            plt.show()
        self.filtering_data_low(self.resp)
        self.filtering_data_high(self.resp_lp)
        self.Result = self.smoothing_data(self.resp_hp)
        return self.resp_sm

    def smoothing_data(self,resp_lp):
        ################################### Smoothing
        # hybrid method
        # step 1
        size= int(1.5*self.Fs)# your code here
        resp_sm0 = scipy.ndimage.uniform_filter1d(resp_lp, size, mode='nearest') # your code here

        # step 2
        # window
        kernel="parzen"
        window = scipy.signal.get_window(kernel, 2*size) # your code here
        w = window / window.sum()

        # Extend signal edges to avoid boundary effects.
        firstvalue = np.repeat(resp_sm0[0], size)
        lastvalue = np.repeat(resp_sm0[-1], size)
        resp_sm0 = np.concatenate((firstvalue, resp_sm0, lastvalue))# your code here
        
        # Compute moving average.
        resp_sm = np.convolve(w, resp_sm0, mode='same')# your code here
        self.resp_sm = resp_sm[size:-size]     
        return self.resp_sm

    def filtering_data_low(self,resp):
        ############ lowpass filtering
        # Parameters
        order = 6 # your code here
        frequency = 0.8 # your code here
        frequency = frequency/(self.Fs/2)  # Normalize frequency to Nyquist Frequency (Fs/2).
        # Filtering
        b, a = scipy.signal.butter(order, frequency, btype='low')
        self.resp_lp = scipy.signal.filtfilt(b, a, resp)
        
        return self.resp_lp

    def filtering_data_high(self, resp_lp):
        ############ lowpass filtering
        # Parameters
        order = 4 # your code here
        frequency = 0.6 # your code here
        frequency = frequency/(self.Fs/2)  # Normalize frequency to Nyquist Frequency (Fs/2).
        # Filtering
        b, a = scipy.signal.butter(order, frequency, btype='high')
        self.resp_hp = scipy.signal.filtfilt(b, a, resp_lp)
        
        return self.resp_hp
    
    
    
    
    
    
    
    