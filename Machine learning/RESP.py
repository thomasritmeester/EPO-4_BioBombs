import numpy as np
import scipy.ndimage
# import os
import numpy as np
import sys


class RESPprep:
    def __init__ (self,Fs,resp_class,title): 
        self.title = title
        self.Fs=Fs
        self.resp = resp_class #[10000:10000+ 10*self.Fs]
        self.t=np.arange(0,self.resp.size*(1/self.Fs),(1/self.Fs))
        self.t=self.t[:self.resp.size]

    def smoothing_data(self,resp, data = 'WESAD'):
        ################################### Smoothing
        # hybrid method
        # step 1
        if(data == 'WESAD'):
            size= int(1.5*self.Fs)# your code here
        else:
            size = int(0.12*self.Fs)# your code here
        resp_sm0 = scipy.ndimage.uniform_filter1d(resp, size, mode='nearest') # your code here

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
        resp_sm = resp_sm[size:-size]

        
        return resp_sm



    def filtering_data(self, resp):
        ############ lowpass filtering
        # Parameters

        order= 5
        freqs=[0.083,0.6]
        frequency = [i * float(2/self.Fs) for i in freqs]
        sos = scipy.signal.butter(order, freqs, btype='bandpass', output="sos", fs= self.Fs)
        resp_bp= scipy.signal.sosfiltfilt(sos, resp)

        
        return resp_bp
