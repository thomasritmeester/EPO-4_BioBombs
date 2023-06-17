# import matplotlib as mpl
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import EDA
import EMG
import ECG
import ACC
import TEMP
import RESP
import DATABASES as data
from DATABASES import WESAD, CSV

# mpl.style.use('seaborn-v0_8')

def label(tekst):
    plt.figure(figsize=(5,1.5))
    text = plt.text(0.5, 0.5, tekst,ha='center', va='center', size=20)
    text.set_path_effects([path_effects.Normal()])
    plt.axis('off')
    plt.show()
    
# Choose Dataset
wesad= False     #Adjust to switch dataset
data.plotting.wesad = wesad     # Tells all plots the stepsize
# Choose if you want to plot
plot = False     #If you want to plot a selection just replace the plot with True or False locally

# Selecting Dataset
if wesad ==True:   
    DATA=WESAD
    dataset ="WESAD"
else:
    DATA=CSV()
    dataset ="C4"

for s in DATA.subject[:]: #looping through your selected subjects -> go to DATABASES.py to vieuw subjects
    label(s) # I use these plots to organises my plots windows in spyder (feel free to comment them)
    
    # importing chosen data
    if wesad ==True:   
        DATA=DATA(s)
        data="WESAD"
    else:
        DATA(s)
        data="C4"

    
    # Importing functions (some signals are missing from certain datasets/subjects)
    # I prefer to make seperate functions for stress and baseline but that's personal preference
    
    if s != "P6":
        EDA_p = EDA.EDAprep(DATA.eda_base,dataset+" baseline: "+str(s))
        EDA_ps = EDA.EDAprep(DATA.eda_stress,dataset+" stress: "+str(s))
        
    EMG_p = EMG.EMGprep(DATA.emg_base,dataset+" baseline: "+str(s))
    EMG_ps = EMG.EMGprep(DATA.emg_stress,dataset+" stress: "+str(s))
    
    ECG_p = ECG.ECGprep(DATA.ecg_base,dataset+" baseline: "+str(s))
    ECG_ps = ECG.ECGprep(DATA.ecg_stress,dataset+" stress: "+str(s))
    
    ACC_p = ACC.ACCprep(DATA.timestamp,dataset+" baseline: "+str(s))
    ACC_ps = ACC.ACCprep(DATA.t_stress,dataset+" stress: "+str(s))
    
    TEMP_p = TEMP.TEMPprep(DATA.timestamp,dataset+" baseline: "+str(s))
    TEMP_ps = TEMP.TEMPprep(DATA.t_stress,dataset+" stress: "+str(s))
    
    if s != "P6":
        RESP_p = RESP.RESPprep(700, DATA.resp_stress, "baseline: "+str(s))
        RESP_ps = RESP.RESPprep(700, DATA.resp_stress, "stress: "+str(s))

    
    # Preprocessing
    #   # EDA
    # label("EDA")
    if s != "P6": 
        EDA_p.alles(plot)
        EDA_ps.alles(plot)
    
    #   # EMG
    # label("EMG")
    EMG_p.alles(plot)
    EMG_ps.alles(plot)
    
    #   # ECG
    # label("ECG")
    ECG_p.alles(plot)
    ECG_ps.alles(plot)
    
    #   # ACC
    # label("ACC")
    if wesad == True:
        ACC_p.alles(DATA.acc_base,plot) 
        ACC_ps.alles(DATA.acc_stress,plot)
    else:
        if s =="P6":
            ACC_p.alles(DATA.ACC1x_base,plot) 
            ACC_ps.alles(DATA.ACC1x_stress,plot)
            
            ACC_p.alles(DATA.ACC1y_base,plot) 
            ACC_ps.alles(DATA.ACC1y_stress,plot)
            
            ACC_p.alles(DATA.ACC1z_base,plot) 
            ACC_ps.alles(DATA.ACC1z_stress,plot)
            
            ACC_p.alles(DATA.ACC2x_base,plot) 
            ACC_ps.alles(DATA.ACC2x_stress,plot)
            
            ACC_p.alles(DATA.ACC2y_base,plot) 
            ACC_ps.alles(DATA.ACC2y_stress,plot)
            
            ACC_p.alles(DATA.ACC2z_base,plot) 
            ACC_ps.alles(DATA.ACC2z_stress,plot)
        else:
            ACC_p.alles(DATA.ACCx_base,plot) 
            ACC_ps.alles(DATA.ACCx_stress,plot)
            
            ACC_p.alles(DATA.ACCy_base,plot) 
            ACC_ps.alles(DATA.ACCy_stress,plot)
            
            ACC_p.alles(DATA.ACCz_base,plot) 
            ACC_ps.alles(DATA.ACCz_stress,plot)
    
    #   # TEMP
    # label("TEMP")
    print(DATA.temp_base)
    TEMP_p.alles(DATA.temp_base,plot)
    TEMP_ps.alles(DATA.temp_stress,plot)
    
    #   # RESP
    if s != "P6":
        # label("RESP")
        RESP_p.alles(plot)
        RESP_ps.alles(plot)
        
    # Getting Pre-processing Results
    # ecg_filt = ECG_p.ecg_filt   # for specific variables just look in the package for the name after the dot
    # ecg_filt_stress = ECG_ps.ecg_filt
    #     # or
    # ecg_filt = ECG_p.Result # the last step outcom has always been coppied to the self.Result, if you used the alles() function
    # ecg_filt = ECG_p.alles(False)
    
        
    #-----------#   #   #   #   "VERSION CONTROL"   #   #   #   #-------------#
    
    # Feature Extraction
    #   # EDA
    # EDA_f = EDA.EDAfeatures(fs)
    # # =======Aan het kieken of ik geen errors krijg================================
    # labels = {'ACC': 4255300, 'ECG': 4255300, 'EMG': 4255300, 'EDA': 4255300, 'Temp': 4255300, 'Resp': 4255300}
    # stress = np.asarray([idx for idx,val in enumerate(labels) if val == 2])
    # t_tot=(len(stress)//(int(0.5*60*fs)))
    # # =============================================================================
    # temp_array = EDA_f.calc_phasic_features(EDA_p.phasic, t_tot, True)
    # #   # ECG
    # ECG_f = ECG.ECGfeatures(ECG_p.ecg_filt) # Import function
    # ECG_f.rpeaks()

    print ("subject "+str(s)+": Plotting...")
    
    from scipy.fft import fft
    fs =700
    
    X = fft(EMG_p.Result)
    freqs = np.arange(0, fs, fs/len(X))
    Xs = fft(EMG_ps.Result)
    freqss = np.arange(0, fs, fs/len(Xs))
    
    plt.figure()
    plt.plot(freqss,abs(Xs)/np.sqrt(len(Xs)), label="Stress", color = "C1")
    plt.plot(freqs,abs(X)/np.sqrt(len(X)), label="Baseline", color = "C0")
    plt.xlim(0-fs/len(X),fs/2)
    plt.title("Filtered EMG signal in frequency domain ("+data+": " +str(s)+")")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("$Magnitude$")
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.plot(freqs,abs(X)/np.sqrt(len(X)), label="Baseline", color = "C0")
    plt.plot(freqss,abs(Xs)/np.sqrt(len(Xs)), label="Stress", color = "C1")
    plt.xlim(0-fs/len(X),fs/2)
    plt.title("Filtered EMG signal in frequency domain ("+data+": " +str(s)+")")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("$Magnitude$")
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.plot(DATA.timestamp,TEMP_p.Result, label="Baseline", color = "C0")
    plt.plot(DATA.t_stress+max(DATA.timestamp),TEMP_ps.Result, label="Stress", color = "C1")
    plt.title("Output Temperature ("+data+": " + str(s)+")")
    plt.xlim(0,max(DATA.timestamp)+max(DATA.t_stress))
    plt.xlim()
    plt.legend()
    plt.xlabel('$Time (s)$') 
    plt.ylabel('$Magnitude$')
    plt.show() 
    
    if s !="P6":
        plt.figure()
        plt.plot(DATA.timestamp,RESP_p.Result, label="Baseline", color = "C0")
        plt.plot(DATA.t_stress+max(DATA.timestamp),RESP_ps.Result, label="Stress", color = "C1")
        plt.title("Output Respration ("+data+": " + str(s)+")")
        plt.xlim(0,max(DATA.timestamp)+max(DATA.t_stress))
        plt.xlim()
        plt.legend()
        plt.xlabel('$Time (s)$') 
        plt.ylabel('$Magnitude$')
        plt.show() 

    print ("subject "+str(s)+": Done")
    
   