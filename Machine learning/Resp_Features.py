from scipy.signal import find_peaks, peak_prominences
import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from RESP import RESPprep


def troughs_and_peaks(function, fs):
    ##Bandpass filter of 0.083 Hz - 0.5 Hz = 5 - 30  BPM

    ##Found all peaks and troughs of the signal
    peaks, _ = find_peaks(function)
    troughs, _ = find_peaks(-function)

    peak_prominence = np.absolute(peak_prominences(function, peaks)[0])
    trough_prominence = np.absolute(peak_prominences(-function, troughs)[0])

    vert_diff = np.hstack((peak_prominence,trough_prominence))

    ##Determine third quartile and threshold: 0.3 * Q3
    
    threshold = 0.2 * np.percentile(vert_diff, 75)

    ##Remove peaks whose difference is smaller than threshold

    for i in range(len(peaks)):
        if (vert_diff[i] < threshold):
            peaks[i] = 0

    for i in range(len(peaks)-2, len(peaks)+len(troughs)):
        if (vert_diff[i] < threshold):
            troughs[i - len(peaks)] = 0


    k = 0
    j = len(peaks)
    i = 0
    while i < j:
        if(peaks[i-k] == 0):
            peaks = np.delete(peaks,i-k)
            k = k + 1
        i = i + 1


    k = 0
    j = len(troughs)
    i = 0
    while i < j:
        if(troughs[i-k] == 0):
            troughs = np.delete(troughs,i-k)
            k = k + 1
        i = i + 1


    # t=np.arange(0,function.size*1/700,1/700)
    # t=t[:function.size]

    # plt.figure(figsize=(14,5))
    # plt.plot(peaks*1/700, function[peaks], "o")
    # plt.plot(troughs*1/700, function[troughs], "o")
    # plt.plot(t, function)
    # plt.title("Respiration Rate Prediction - Baseline Wandering")
    # plt.xlabel("Time $(s)$")
    # plt.ylabel("Amplitude")
    # plt.show()
    
    return troughs, peaks



def calc_resp_features(resp_data, fs, frame):

    resp_features = pd.DataFrame()


    wdw=int(frame*fs)
    size_adpt=(int(wdw))

    #print(size_adpt, "size of the samples")
    t_tot = int(len(resp_data)//(int(wdw)))
    resp_tot = np.zeros([size_adpt, t_tot])
    #ecg_base_tot = np.zeros([size_adpt, t_tot])
    #print(t_tot, 'ttot1')

    RESP = []

    for i in range(t_tot):
        RESP = RESPprep(fs,resp_data, "")
        resp_filt = RESP.filtering_data(resp_data)
        resp1 = resp_filt[i*int(wdw):(i+1)*int(wdw)]
        #ecg2 = base[i*int(wdw):(i+1)*int(wdw)]
        # t1 = np.arange(0, len(resp1)*(1/fs), (1/fs))
        # t1 = t1[:len(resp1)]
        # t2 = np.arange(0, ecg2.size*(1/fs), (1/fs))
        # t2 = t2[:ecg2.size]
        resp_tot[:, i] = resp1


        troughs, peaks = troughs_and_peaks(resp_tot[:, i], fs)

        features = np.zeros(6)

        t_intervals = np.asarray([])

        for i in range(1,len(peaks)):
            t_intervals = np.append(t_intervals, (peaks[i] - peaks[i-1])/fs)
        
        RR = 60/np.mean(t_intervals)


        if(troughs[0] < peaks[0]):
            inhalation = (peaks[:min(len(troughs), len(peaks))] - troughs[:min(len(troughs), len(peaks))])/fs
            exhalation = (troughs[1:min(len(troughs), len(peaks))] - peaks[:min(len(troughs), len(peaks))-1])/fs
        
        else:
            inhalation = (peaks[1:min(len(troughs), len(peaks))] - troughs[:min(len(troughs), len(peaks))-1])/fs
            exhalation = (troughs[:min(len(troughs), len(peaks))] - peaks[:min(len(troughs), len(peaks))])/fs

        in_mean = np.mean(inhalation)
        in_std = np.std(inhalation)

        ex_mean = np.mean(exhalation)
        ex_std = np.std(exhalation)

        resp_rmssd = np.sqrt(np.mean([j**2 for j in t_intervals]))

        features[0] = RR
        features[1] = in_mean
        features[2] = in_std
        features[3] = ex_mean
        features[4] = ex_std
        features[5] = resp_rmssd

        pd_features = pd.DataFrame([features], columns = ['Respiration Rate', 'Inahaltion Mean', 'Inahaltion Std', 'Exhalation Mean', 'Exhalation Std', 'RMSSD'])

        resp_features = pd.concat([resp_features, pd_features], ignore_index=True)

    return resp_features
        
