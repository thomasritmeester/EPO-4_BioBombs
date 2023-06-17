import matplotlib.pyplot as plt
import numpy as np
import time

class DynamicUpdate():
    #Suppose we know the x range
    min_x = 0       # tijd as begin
    max_x = 10*60   # tijd as einde
    
    def __init__(self, Fs):
        self.Fs = Fs

    def on_launch(self):
        #Set up plot
        plt.ion()
        self.figure, self.ax = plt.subplots()
        self.lines, = self.ax.plot([],[])
        #Autoscale on unknown axis and known lims on the other
        self.ax.set_autoscaley_on(True)
        self.ax.set_xlim(self.min_x, self.max_x)
        #Other stuff
        self.ax.grid()
        ...
        self.xdata = []
        self.ECGdata = []

    def on_running(self, xdata, ydata):
        #Update data (with the new _and_ the old points)
        self.lines.set_xdata(xdata)
        self.lines.set_ydata(ydata)
        #Need both of these in order to rescale
        self.ax.relim()
        self.ax.autoscale_view()
        #We need to draw *and* flush
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    #Example
    def go(self,x,ECG):
        
        self.xdata.append(x)
        self.ECGdata.append(ECG)
        self.on_running(self.xdata, self.ECGdata)
            
        return self.xdata, self.ECGdata

# d = DynamicUpdate()
# d(x,y)
