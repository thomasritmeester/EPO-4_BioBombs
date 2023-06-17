import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import time
testoutput=[1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1,1, 1, 0, 0,0, 0,0]
class DynamicUpdate():
    min_x = 0       
    max_x = 10*60
    
   # def __init__(self, Fs):
        #self.Fs = Fs

    def on_launch(self, MLoutput):
        #Set up plot
        plt.ion()
        labels = ['no stress', 'stress']
        ys = [0, 1]
        self.figure, self.ax = plt.subplots()
        self.lines, = self.ax.plot([],[],)
        self.ax.axhline(y = 0.5, color = 'r', linestyle = '--')
        #Autoscale on unknown axis and known lims on the other
        #self.ax.set_autoscaley_on(True)
        self.ax.set_autoscaley_on(True)
        self.ax.set_xlim(0, len(MLoutput))
        self.ax.set_ylim(-0.3, 1.5)
        #Other stuff
        self.ax.grid()
        #self.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        self.ax.set_yticks(ys, labels) # set labesl to stress and no stress
        self.ax.text(len(MLoutput)/2, 1.2, 'stress', size=15, bbox=dict(boxstyle="round", ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8),))
        self.ax.text(len(MLoutput)/2, -0.2, 'no stress', size=15, bbox=dict(boxstyle="round", ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8),))
        ...

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
    def __call__(self, MLoutput):
        self.on_launch(MLoutput)
        xdata = []
        ydata = []
        for self.x in np.arange(0,len(MLoutput),1):
            xdata.append(self.x)
            ydata.append(MLoutput[self.x])

            self.on_running(xdata, ydata)
            time.sleep(1)
        return xdata, ydata

d = DynamicUpdate()
d(testoutput)