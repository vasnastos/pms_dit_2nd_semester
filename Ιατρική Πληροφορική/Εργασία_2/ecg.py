import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class ECG:
    def __init__(self) -> None:
        self.signal=np.loadtxt('ecg.txt')
        self.sliced_signal=self.signal[100:1000]
    
    def plot_signal(self):
        plt.figure(figsize=(10,3))
        plt.plot(np.arange(self.signal.shape[0]),self.signal)
        plt.ylim(self.signal.min()-0.5,self.signal.max()+0.5)
        plt.xticks(np.arange(0,self.signal.shape[0],100))
        ax=plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.show()
    
    def plot_sliced_signal(self):
        plt.figure(figsize=(10,3))
        plt.plot(np.arange(self.sliced_signal.shape[0]),self.sliced_signal)
        plt.ylim(self.sliced_signal.min()-0.5,self.sliced_signal.max()+0.5)
        plt.xticks(np.arange(0,self.sliced_signal.shape[0],100))
        ax=plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.show()
    
    def amplitude_range(self):
        # Calculate DFT of the Signal
        frequency=50000 # HZ
        
        dft_signal=np.fft.fft(self.sliced_signal)
        ampitude_spectrum=2*np.abs(dft_signal)/dft_signal.shape[0]
        frq_amp_range=np.fft.fftfreq(ampitude_spectrum.shape[0],1/frequency)
        
        
        plt.figure(figsize=(10,5))
        plt.scatter(np.arange(frq_amp_range.shape[0]),frq_amp_range,color='r',marker='o')
        # for i in range(frq_amp_range.shape[0]):
        #     plt.plot([0,i],[0,frq_amp_range[i]])  
          
        plt.ylim(frq_amp_range.min()-0.5,frq_amp_range.max()+0.5)
        plt.xticks(np.arange(0,frq_amp_range.shape[0],100))
        ax=plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.show()



if __name__=='__main__':
    ecg=ECG()
    ecg.amplitude_range()