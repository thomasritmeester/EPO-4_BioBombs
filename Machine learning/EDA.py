# import os
# from os import listdir
# from os.path import isfile, join, isdir
import pickle
import numpy as np
# from scipy import signal as sig
# import numpy as np
#import matplotlib.pyplot as plt
import sys
import scipy.ndimage
#sys.path.append("/usr/local/lib/python3.7/site-packages")

class EDAprep:
    def __init__ (self,Fs,eda_class,title):
        self.title = title
        self.Fs=Fs
        self.eda = eda_class #[10000:10000+ minutes*60*self.Fs]  
        self.t=np.arange(0,self.eda.size*(1/self.Fs),(1/self.Fs))
        self.t=self.t[:self.eda.size]
        
    def plotdata(self,title=""):
        # cut a smaller window      
        if title=="":
            title=self.title
        '''
        plt.title("input EDA "+ title)
        plt.plot(self.t,self.eda)
        plt.xlabel('$Time (s)$') 
        plt.ylabel('$EDA$')
        plt.show()'''
    
    def filtering_data(self,title=""):
        ############ lowpass filtering
        # Parameters
        order = 4 # your code here
        frequency = 5 # your code here
        frequency = frequency/(self.Fs/2)  # Normalize frequency to Nyquist Frequency (Fs/2).
        # Filtering
        b, a = scipy.signal.butter(order, frequency, btype='low')
        eda_lp = scipy.signal.filtfilt(b, a, self.eda)
        
        '''# plot
        if title=="":
            title=self.title
        plt.title("EDA lowpass filtered "+ title)
        plt.plot(self.t,self.eda)
        plt.plot(self.t,eda_lp)
        plt.ylim(np.min(eda_lp)-1,np.max(eda_lp)+1)
        # labels and titles
        plt.xlabel('$Time (s)$') 
        plt.ylabel('$EDA$') 
        plt.show()'''
        return eda_lp
    
    def smoothing_data(self,eda_lp, title=""):
        ################################### Smoothing
        # hybrid method
        # step 1
        size= int(0.75*self.Fs)# your code here
        eda_sm0 = scipy.ndimage.uniform_filter1d(eda_lp, size, mode='nearest') # your code here

        # step 2
        # window
        kernel="parzen"
        window = scipy.signal.get_window(kernel, 2*size) # your code here
        w = window / window.sum()

        # Extend signal edges to avoid boundary effects.
        firstvalue = np.repeat(eda_sm0[0], size)
        lastvalue = np.repeat(eda_sm0[-1], size)
        eda_sm0 = np.concatenate((firstvalue, eda_sm0, lastvalue))# your code here
        # eda_sm0[0:19] = eda_sm0[20]
        
        # Compute moving average.
        eda_sm = np.convolve(w, eda_sm0, mode='same')# your code here
        # eda_sm = eda_sm[:len(self.eda)]
        eda_sm = eda_sm[size:-size]

        
        '''# plot
        if title=="":
            title=self.title
        plt.title("EDA smoothed "+ title)
        plt.plot(self.t,self.eda)
        plt.plot(self.t,eda_sm)
        #plt.xlim(70, 90)
        # labels and titles
        plt.xlabel('$Time (s)$') 
        plt.ylabel('$EDA$') 
        plt.show()'''
        
        return eda_sm
    
    def decompose_data(self,eda_sm,title=""):
        ################## decompose 
        # Electrodermal Activity (EDA) into Phasic and Tonic components.
        # Phasic
        order=5
        freqs=[0.05]
        sos = scipy.signal.butter(order, freqs, btype='high', output="sos", fs= self.Fs)
        phasic= scipy.signal.sosfiltfilt(sos, eda_sm)
        
        #Tonic
        order=5
        freqs=[0.05]
        sos = scipy.signal.butter(order, freqs, btype='low', output="sos",  fs= self.Fs)
        tonic= scipy.signal.sosfiltfilt(sos, eda_sm)
        
        '''# plot
        if title=="":
            title=self.title
        plt.title("EDA meer "+ title)
        plt.plot(self.t,eda_sm,label='smooth_eda')
        plt.plot(self.t,phasic,label='phasic')
        plt.plot(self.t,tonic,label='tonic')
        # labels and titles
        plt.xlabel('$Time (s)$') 
        plt.ylabel('$EDA$') ()
        plt.legend()
        plt.show()'''
        
        return eda_sm, phasic, tonic

'''
EDA = {}
#EDA = EDAprep(eda_base,20,"baseline")
   
EDA.plotdata()
eda_lp = EDA.filtering_data()
eda_sm = EDA.smoothing_data(eda_lp)
EDA.decompose_data(eda_sm)

EDA = {}
EDA = EDAprep(eda_stress,10,"stress")

EDA.plotdata()
eda_lp = EDA.filtering_data()
eda_sm = EDA.smoothing_data(eda_lp)
EDA.decompose_data(eda_sm)

EDA = {}
EDA = EDAprep(eda_amusement,20,"amusement")

EDA.plotdata()
eda_lp = EDA.filtering_data()
eda_sm = EDA.smoothing_data(eda_lp)
EDA.decompose_data(eda_sm)

EDA = {}
EDA = EDAprep(eda_meditation,20,"meditation")

EDA.plotdata()
eda_lp = EDA.filtering_data()
eda_sm = EDA.smoothing_data(eda_lp)
EDA.decompose_data(eda_sm)  
'''
