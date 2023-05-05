import numpy as np,os
import matplotlib.pyplot as plt

class ECG:
    def __init__(self) -> None:
        self.signal=np.loadtxt('ecg.txt')
        self.sliced_signal=self.signal[100:1000]
        self.frequency=1000
        self.cutoff_freq=None
        self.fig=plt.figure(figsize=(10,6))
        self.rows=4
        self.cols=2
        self.index=1

    def plot_signal(self,in_axis=False):
        if not in_axis:
            plt.figure(figsize=(15,6))
            plt.plot(np.arange(self.signal.shape[0]),self.signal)
            plt.ylim(self.signal.min(),self.signal.max())
            plt.xticks(np.arange(0,self.signal.shape[0],5000))
            plt.xlabel("n(samples)")
            plt.ylabel('ecg')
            ax=plt.gca()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.title('Full signal plot')
            plt.savefig(os.path.join('','figures','ecg_signal_full.png'),dpi=300)
            plt.show()
        
        else:
            ax=self.fig.add_subplot(self.rows,self.cols,self.index,colspan=2)
            ax.plot(np.arange(self.signal.shape[0]),self.signal)
            ax.set_ylim(self.signal.min(),self.signal.max())
            ax.set_xticks(np.arange(0,self.signal.shape[0],5000))
            ax.set_xlabel("n(samples)")
            ax.set_ylabel('ecg')
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.set_title('Full signal plot')
            self.index+=2
    
    def plot_sliced_signal(self,in_axis=False):
        if not in_axis:
            plt.figure(figsize=(10,3))
            plt.plot(np.arange(self.sliced_signal.shape[0]),self.sliced_signal)
            plt.ylim(self.sliced_signal.min()-0.5,self.sliced_signal.max()+0.5)
            plt.xticks(np.arange(0,self.sliced_signal.shape[0],100))
            plt.xlabel("n(samples)")
            plt.ylabel('ecg')
            ax=plt.gca()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.title('Sliced signal(100:1000) plot')
            plt.savefig(os.path.join('','figures','ecg_sliced_signal_100_1000.png'),dpi=300)
            plt.show()
        else:
            ax=self.fig.add_subplot(self.rows,self.cols,self.index)
            ax.plot(np.arange(self.sliced_signal.shape[0]),self.sliced_signal)
            ax.set_ylim(self.sliced_signal.min()-0.5,self.sliced_signal.max()+0.5)
            ax.set_xticks(np.arange(0,self.sliced_signal.shape[0],100))
            ax.set_xlabel("n(samples)")
            ax.set_ylabel('ecg')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_title('Sliced signal(100:1000) plot')
            self.index+=1
        

    def amplitude_range(self,in_axis=False):
        # Fasma megethous
        windowed_ecg_signal=np.hanning(len(self.sliced_signal))*self.sliced_signal
        fft_signal=np.fft.fft(windowed_ecg_signal)
        self.magnitude_spectrum=2.0/len(self.sliced_signal)*np.abs(fft_signal)

        sampling_rate=1000 #Hz
        self.frequency_axis=np.linspace(0,sampling_rate,len(self.magnitude_spectrum))
        if not in_axis:
            plt.figure(figsize=(10,6))
            plt.plot(self.frequency_axis,self.magnitude_spectrum)
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Ampitude')

            ax=plt.gca()
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            plt.title('Magnitude spectrum signal plot')
            plt.savefig(os.path.join('','figures','magnitude_spectrum_fft_signal.png'),dpi=300)
            plt.show()
        else:
            ax=self.fig.add_subplot(self.rows,self.cols,self.index)
            ax.plot(self.frequency_axis,self.magnitude_spectrum)
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('Ampitude')

            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.set_title('Magnitude spectrum signal plot')
            self.index+=1


    def calculate_snr(self,signal,noise,cutoff_freq):
        signal_spectrum=2.0/len(signal)*np.abs(np.fft.fft(signal))
        noise_spectrum=2.0/len(noise)*np.abs(np.fft.fft(noise))

        freqs=np.fft.fftfreq(len(signal),d=1/self.frequency)
        freq_mask=freqs<=cutoff_freq
        signal_power=np.sum(signal_spectrum[freq_mask]**2)
        noise_power=np.sum(noise_spectrum[freq_mask]**2)

        return signal_power/noise_power

    def cutoff_frequency(self,in_axis=False):
        noise=np.random.normal(0,0.5,self.sliced_signal.shape[0])

        cutoff_freqs=np.linspace(0,self.frequency,num=100)
        snrs=[self.calculate_snr(signal=self.sliced_signal,noise=noise,cutoff_freq=cutoff_freq) for cutoff_freq in cutoff_freqs]

        best_cutoff_idx=np.argmax(snrs)
        self.cutoff_freq=cutoff_freqs[best_cutoff_idx]

        max_indeces=np.array(snrs).argsort()[-2:]
        if not in_axis:
            plt.figure(figsize=(10,5))
            plt.plot(self.frequency_axis,self.magnitude_spectrum)  
            plt.axvline(x=cutoff_freqs[max_indeces[0]], color='red', linestyle='--', label=f'Cutoff Frequency = {cutoff_freqs[max_indeces[0]]:.2f} Hz')
            plt.ylim(self.magnitude_spectrum.min(),self.magnitude_spectrum.max()+0.5)
            plt.xlim(0,self.frequency_axis.max()+2)
            plt.xticks(np.arange(0,self.frequency_axis.max(),100))
            ax=plt.gca()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.legend()
            plt.title('Cutoff frequency plot')
            plt.savefig(os.path.join('','figures','cutoff_frequency(frequency_spectrum)_signal.png'),dpi=300)
            plt.show()
        else:
            ax=self.fig.add_subplot(self.rows,self.cols,self.index)
            ax.plot(self.frequency_axis,self.magnitude_spectrum)  
            ax.axvline(x=cutoff_freqs[max_indeces[0]], color='red', linestyle='--', label=f'Cutoff Frequency = {cutoff_freqs[max_indeces[0]]:.2f} Hz')
            ax.set_ylim(self.magnitude_spectrum.min(),self.magnitude_spectrum.max()+0.5)
            ax.set_xlim(0,self.frequency_axis.max()+2)
            ax.set_xticks(np.arange(0,self.frequency_axis.max(),100))
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.legend()
            ax.set_title('Cutoff frequency plot')
            self.index+=1
    
    def zero_high_freq_components(self,in_axis=False):
        fft_signal=np.fft.fft(self.sliced_signal)
        indices = np.where(np.abs(np.fft.fftfreq(self.sliced_signal.size, d=1/len(self.sliced_signal))) > self.cutoff_freq)[0]

        fft_signal[indices]=0
        fft_signal[-indices]=0

        filtered_signal=np.fft.ifft(fft_signal).real
       
        if not in_axis:
            plt.figure(figsize=(10,5))
            plt.plot(2.0/len(filtered_signal)*np.abs(np.fft.fft(filtered_signal)),color='r')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Amplitude')
            plt.title('Filtered Spectrum')
            ax=plt.gca()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.savefig(os.path.join('','figures','zero_high_freq_components.png'),dpi=300)
            plt.show()
        else:
            ax=self.fig.add_subplot(self.rows,self.cols,self.index)
            ax.plot(2.0/len(filtered_signal)*np.abs(np.fft.fft(filtered_signal)),color='r')
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('Amplitude')
            ax.set_title('Filtered Spectrum')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    def signal_reconstruction(self,in_axis=False):
        ecg_fft=np.fft.fft(self.sliced_signal)
        indeces=np.where(np.abs(np.fft.fftfreq(self.sliced_signal.size, d=1/len(self.sliced_signal))) > self.cutoff_freq)[0]

        ecg_fft[indeces]=0
        ecg_fft[-indeces]=0
        filtered_signal=np.fft.ifft(ecg_fft).real

        plt.figure(figsize=(10,5))
        plt.plot(np.linspace(0,1,len(self.sliced_signal)),filtered_signal)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        ax=plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.title('Reconstructed Signal')
        plt.savefig(os.path.join('','figures','signal_reconstruction.png'),dpi=300)
        plt.show()

    def show(self):
        self.plot_signal(in_axis=True)
        self.plot_sliced_signal(in_axis=True)
        self.amplitude_range(in_axis=True)
        self.cutoff_frequency(in_axis=True)
        self.zero_high_freq_components(in_axis=True)
        self.signal_reconstruction(in_axis=True)

if __name__=='__main__':
    ecg=ECG()
    ecg.plot_signal()
    ecg.plot_sliced_signal()
    ecg.amplitude_range()
    ecg.cutoff_frequency()
    ecg.zero_high_freq_components()
    ecg.signal_reconstruction()