import numpy as np
import neurokit2 as nk
import pandas as pd

# Download example data
data = nk.data("bio_eventrelated_100hz")
print(data)

# Preprocess the data (filter, find peaks, etc.)
processed_data, info = nk.bio_process(ecg=data["ECG"], rsp=data["RSP"], eda=data["EDA"], sampling_rate=100)

# Compute relevant features
results = nk.bio_analyze(processed_data, sampling_rate=100)

ecg = nk.ecg_simulate(duration=10, heart_rate=70)

# Visualise biosignals
data = pd.DataFrame({"ECG": ecg})
print(data)
nk.signal_plot(data)
