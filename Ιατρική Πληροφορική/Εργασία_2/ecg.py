import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy


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
        frequency=1000 # HZ
        t=np.linspace(0,1,frequency)


        dft_signal=np.fft.fft(self.sliced_signal)
        ampitude_spectrum=2.0/self.sliced_signal.shape[0] * np.abs(dft_signal)
        frq_amp_range=np.fft.fftfreq(ampitude_spectrum.shape[0],1/frequency)
        
        
        plt.figure(figsize=(10,5))
        plt.plot(frq_amp_range,ampitude_spectrum)  
        plt.ylim(ampitude_spectrum.min(),ampitude_spectrum.max()+0.5)
        plt.xlim(0,frq_amp_range.max()+2)
        plt.xticks(np.arange(0,frq_amp_range.shape[0],100))
        ax=plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.show()

    def calculate_snr(self,signal,noise,fs,cutoff_freq):
        signal_spectrum=np.abs(np.fft.fft(signal))
        noise_spectrum=np.abs(np.fft.fft(noise))

        freqs=np.fft.fftfreq(len(signal),d=1/fs)
        freq_mask=freqs<=cutoff_freq
        signal_power=np.sum(signal_spectrum[freq_mask]**2)
        noise_power=np.sum(noise_spectrum[freq_mask]**2)

        return signal_power/noise_power

    def cutoff_frequency(self):
        fs=1000
        t=np.linspace(0,1,1000)
        noise=np.random.normal(0,0.5,self.sliced_signal.shape[0])

        cutoff_freqs=np.linspace(0,fs/2,num=100)
        snrs=[self.calculate_snr(signal=self.sliced_signal,noise=noise,fs=fs,cutoff_freq=cutoff_freq) for cutoff_freq in cutoff_freqs]

        best_cutoff_idx=np.argmax(snrs)
        best_cutoff_freq=cutoff_freqs[best_cutoff_idx]
        print(best_cutoff_idx,best_cutoff_freq)


if __name__=='__main__':
    ecg=ECG()
    ecg.amplitude_range()
    ecg.cutoff_frequency()