import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
from scipy.signal import chirp, find_peaks, peak_widths, peak_prominences
from DATABASES import plotting as An


class EDAprep:
    Fs=700
    if An.wesad == False:
        g=1000000
    else:
        g= Fs
    def __init__ (self,eda_class=[],title=""):
        self.title = title
        self.eda = eda_class
        self.plot=False
        self.t=np.arange(0,self.eda.size*(1/self.g),(1/self.g))
        self.t=self.t[:self.eda.size]
        
    def alles(self, plot=False):
        self.plot=plot
        if plot == True:
            self.plotdata()
        self.filtering_data()
        self.smoothing_data(self.eda_lp)
        self.Result = self.decompose_data(self.smooth)
        return self.phasic
       
        
    def plotdata(self,title=""):
        # cut a smaller window      
        if title=="":
            title=self.title
        plt.figure(figsize=(10,4))
        plt.title("input EDA ("+title+")")
        plt.plot(self.t,self.eda)
        plt.xlabel('$Time (s)$') 
        plt.xlim(0,max(self.t))
        plt.ylabel('$Magnitude$')
        plt.show()
    
    def filtering_data(self,title=""):
        ############ lowpass filtering
        # Parameters
        order = 4 # your code here
        frequency = 5 # your code here
        frequency = frequency/(self.Fs/2)  # Normalize frequency to Nyquist Frequency (Fs/2).
        # Filtering
        b, a = scipy.signal.butter(order, frequency, btype='low')
        eda_lp = scipy.signal.filtfilt(b, a, self.eda)
        self.eda_lp= eda_lp
        
        # # plot
        if self.plot == True:
            
            if title=="":
                title=self.title
            plt.figure(figsize=(10,4))
            plt.title("EDA lowpass filtered ("+title+")")
            plt.plot(self.t,self.eda)
            plt.plot(self.t,eda_lp)
            plt.ylim(np.min(eda_lp)-1,np.max(eda_lp)+1)
            plt.xlim(0,max(self.t))
            # labels and titles
            plt.xlabel('$Time (s)$') 
            plt.ylabel('Magnitude') 
            plt.show()
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
        self.smooth = eda_sm

        if self.plot == True:
            
            # plot
            if title=="":
                title=self.title
            plt.figure(figsize=(7,4))
            plt.title("EDA smoothed ("+title+")")
            plt.plot(self.t,self.eda)
            plt.plot(self.t,eda_sm)
            plt.xlim(0,max(self.t))
            # labels and titles
            plt.xlabel('$Time (s)$') 
            plt.ylabel('$Magnitude$') 
            plt.show()
        
        return eda_sm
    
    def decompose_data(self,eda_sm,title=""):
        ################## decompose 
        # Electrodermal Activity (EDA) into Phasic and Tonic components.
        # Phasic
        order=5
        freqs=[0.05]
        sos = scipy.signal.butter(order, freqs, btype='high', output="sos", fs= self.Fs)
        phasic= scipy.signal.sosfiltfilt(sos, eda_sm)
        self.phasic = phasic
        
        #Tonic
        order=5
        freqs=[0.05]
        sos = scipy.signal.butter(order, freqs, btype='low', output="sos",  fs= self.Fs)
        tonic= scipy.signal.sosfiltfilt(sos, eda_sm)
        self.tonic = tonic
        
        if self.plot == True:
            
            # plot
            if title=="":
                title=self.title
            plt.figure(figsize=(7,4))
            plt.title("EDA signals ("+title+")")
            plt.plot(self.t,eda_sm,label='smooth_eda')
            plt.plot(self.t,phasic,label='phasic')
            plt.plot(self.t,tonic,label='tonic')
            plt.xlim(0,max(self.t))
            # labels and titles
            plt.xlabel('$Time (s)$') 
            plt.ylabel('$Magnitude$') 
            plt.legend()
            plt.show()
        
        return eda_sm, phasic, tonic
    
    def eda_phasic(self):
        return self.phasic
    
    def eda_tonic(self):
        return self.tonic
    
class EDAfeatures:
    def __init__(self,Fs):
        self.Fs = Fs
        
    def calc_phasic_data(self,phasic, peak, height):
        ##Find all the points on the plot below the 50% of the peak
        half_points = np.where(((phasic - (phasic[peak] - 0.5*height) < 0.00001)))[0]
        
        ##Finds the index of the point directly to the right of the peak
        half_amp_index = np.inf
        for j in half_points:
            if(j < half_amp_index and j > peak):
                half_amp_index = j

        ##Calculate onset and offset
        onset_points = np.where(((phasic - (phasic[peak] - 0.63*height) < 0.00001)))[0]
        
        ##Finds the index of the point directly to the right of the peak
        offset_amp_index = np.inf
        for j in onset_points:
            if(j < offset_amp_index and j > peak):
                offset_amp_index = j
        
        ##Finds the index of the point directly to the right of the peak
        onset_amp_index = 0
        for j in onset_points:
            if(j > onset_amp_index and j < peak):
                onset_amp_index = j
        return onset_amp_index, half_amp_index, offset_amp_index 
    
    def calc_phasic_features(self, phasic, t_tot, state):
  
        temp_array = np.asarray([], dtype = "float")
        
        
        orienting_mag = np.asarray([], dtype = "float")
        orienting_time = np.asarray([], dtype = "float")
        half_recov_time = np.asarray([], dtype = "float")
        
        peaks, _ = find_peaks(phasic)
        heights, _, __ = peak_prominences(phasic, peaks)
        widths, _, __, ___ = peak_widths(phasic, peaks, rel_height=0.63)
        
        # find the indices with an amplitude larger that 0.1
        keep = np.full(len(peaks), True)
        keep[peaks < 0.1] = False
        
        # only keep those
        peaks=peaks[keep]
        heights=heights[keep]
        widths=widths[keep]
        
        
        for i in range(len(peaks)):  
            onset_amp_index, half_amp_index, offset_amp_index = self.calc_phasic_data(phasic, peaks[i], heights[i])
            
            orienting_mag = np.append(orienting_mag, phasic[peaks[i]] - phasic[onset_amp_index])
            orienting_time = np.append(orienting_time, (offset_amp_index - onset_amp_index)/self.Fs)
            half_recov_time = np.append(half_recov_time, (half_amp_index - peaks[i])/self.Fs)
        
        cv_mg = np.std(orienting_mag)/np.mean(orienting_mag)
        cv_orient = np.std(orienting_time)/np.mean(orienting_time)
        cv_recov = np.std(half_recov_time)/np.mean(half_recov_time)
        
        temp_array = np.append(temp_array, cv_mg)
        temp_array = np.append(temp_array, cv_orient)
        temp_array = np.append(temp_array, cv_recov)
        
        if(state == False):
            temp_array = np.append(temp_array,0)
        else:
            temp_array = np.append(temp_array,1)

        return temp_array

# EDA = {}
# EDA = EDAprep(fs,eda_base,20,"baseline")
   
# EDA.plotdata() 

# eda_lp = EDA.filtering_data()
# eda_sm = EDA.smoothing_data(eda_lp)
# EDA.decompose_data(eda_sm)

