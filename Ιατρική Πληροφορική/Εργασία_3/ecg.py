import os,numpy as np,statistics
import matplotlib.pyplot as plt
from tabulate import tabulate
from scipy.stats import skew,kurtosis


def dft(signal):
    N=len(signal)

    RFx=np.zeros(N)
    IFx=np.zeros(N)

    for k in range(N):
        for n in range(N):
            RFx[k]+=signal[n] * np.cos(2 * np.pi * k * n/N)
            IFx[k]-=signal[n] * np.sin(2 * np.pi * k * n/N)
    return RFx,IFx

def idft(RFx,IFx):
    N=len(RFx)

    Ry=np.zeros(N)
    Iy=np.zeros(N)

    for n in range(N):
        for k in range(N):
            Ry[n]+=RFx[k] * np.cos(2 * np.pi * k * n / N)
            Iy[n]+=IFx[k] * np.sin(2 * np.pi * k * n /N)
    return Ry,Iy

def convolution_freq(x,y):
    N = max(len(x), len(y))
    x_padded = np.pad(x, (0, N - len(x)), mode='constant')
    h_padded = np.pad(y, (0, N - len(y)), mode='constant')

    X = np.fft.fft(x_padded)
    H = np.fft.fft(h_padded)

    y= X * H
    return np.real(np.fft.ifft(y))


def correlation(x, y):
    # Pad the signals with zeros to match their lengths
    N = max(len(x), len(y))
    x_padded = np.pad(x, (0, N - len(x)), mode='constant')
    y_padded = np.pad(y, (0, N - len(y)), mode='constant')
    
    # Calculate correlation using the convolution of x and reversed y
    z = np.flip(y_padded)  # Reverse the second signal
    r = np.convolve(x_padded, z)
    
    return r


class ECG:
    def __init__(self) -> None:
        self.ecg_signal=np.loadtxt(os.path.join('','ecg.txt'))
        self.sliced_signal=self.ecg_signal[50:300]
    
    def plot_signal(self):
        plt.figure(figsize=(10,6))
        plt.plot(np.arange(len(self.sliced_signal)),self.sliced_signal,color='r')
        ax=plt.gca()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.savefig(os.path.join('','figures','singal.png'))
        plt.show()
    
    def statistics(self):
        q1,q3,_=statistics.quantiles(data=self.sliced_signal,n=4)
        rows=[
            ['Samples',self.sliced_signal.shape[0]],
            ['Mean',statistics.mean(self.sliced_signal)],
            ['Median',statistics.median(self.sliced_signal)],
            ['Std',statistics.stdev(self.sliced_signal)],
            ['Iqr',q3-q1],
            ['Skewness',skew(self.sliced_signal)],
            ['Kurtosis',kurtosis(self.sliced_signal)]
        ]
        print(tabulate(tabular_data=rows,headers=['Statistic Meter','Value'],tablefmt='fancy_grid',floatfmt='.3f'))
        with open(os.path.join('stats.tex'),'w') as writer:
            writer.write(tabulate(tabular_data=rows,headers=['Statistic Meter','Value'],tablefmt='latex',floatfmt='.3f'))


    def normalize_carediogram(self,shock_response):
        signal_fft=np.fft.fft(self.sliced_signal)
        shock_response_fft=np.fft.fft(shock_response,len(self.sliced_signal))
        smoothed_signal_fft=signal_fft * shock_response_fft
        return np.real(np.fft.ifft(smoothed_signal_fft))

    def congruence_in_the_frequency_domain(self,shock_response):
        signal_copy=self.sliced_signal
        if len(self.sliced_signal)<len(shock_response):
            signal_copy=np.pad(signal_copy,(0,len(shock_response)-len(signal_copy)),'constant')
        elif len(self.sliced_signal)>len(shock_response):
            shock_response=np.pad(shock_response,(0,len(self.sliced_signal)-len(shock_response)),'constant')

        N=len(signal_copy)
        signal_copy=np.concatenate((np.zeros(N),signal_copy))
        shock_response=np.concatenate((shock_response,np.zeros(N)))

        RFx,IFx=dft(signal_copy)
        RFsh,IFsh=dft(shock_response)

        # Γινόμενο των DFT (Εύρεση συνέλιξης στο πεδίο των συχνοτήτων)
        RFy=RFx * RFsh - IFx * IFsh
        IFy= RFx * IFsh + IFx *RFsh

        Ry,Iy=idft(RFy,IFy)
        freq_signal=len(signal_copy) * Ry
        return freq_signal
    
    def plot_sliced_and_normalized(self):
        normalized_cardiogram=self.normalize_carediogram(np.array([0.25,0.5,0.25]))
        plt.figure(figsize=(10,6))
        plt.plot(np.arange(len(self.sliced_signal)),self.sliced_signal,color='b')
        plt.plot(np.arange(len(normalized_cardiogram)),normalized_cardiogram,color='r')
        ax=plt.gca()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.xticks(np.arange(0,300,20))
        plt.savefig(os.path.join('','figures','normalized_cardiogram.png'))
        plt.show()
    
    def plot_sliced_and_correlated(self):
        shifted_signal=self.congruence_in_the_frequency_domain(np.array([0.25,0.5,0.25]))
        plt.figure(figsize=(10,6))
        plt.plot(np.arange(len(self.sliced_signal)),self.sliced_signal,color='b')
        plt.plot(np.arange(len(shifted_signal)),shifted_signal,color='r')
        ax=plt.gca()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # plt.xticks(np.arange(0,300,20))
        plt.savefig(os.path.join('','figures','shifted_cardiogram.png'))
        plt.show()
#                           Answer-Question 2
# When doubling the length of the two signals by multiplying by zero and padding with zeros, the signals need to be shifted to ensure correct convolution using the correlation theorem.
# By convention, when performing convolution in the time domain, the output signal is obtained by shifting one of the signals and flipping it. This is necessary to 
# align the signals properly for the multiplication and summation operations.
# When doubling the length of the signals, one signal is incremented to the right by padding zeros at the beginning. This rightward shift ensures that when the signals are flipped during convolution, the proper alignment is maintained with the other signal.
# On the other hand, the other signal is incremented to the left by padding zeros at the end. This leftward shift ensures that after flipping, the signal is correctly aligned with the shifted signal during convolution.
# In summary, the leftward and rightward shifts of the signals when doubling their length with zeros ensure that the signals are properly aligned for correct convolution using the correlation theorem.

    def plot_sliced_convolution_freq(self):
        shock_response=np.array([0.25,0.5,0.25])
        y_freq=convolution_freq(self.sliced_signal,shock_response)
        y_corr=correlation(self.sliced_signal,shock_response)

        print('Output from convolution freq:')
        print(y_freq,end='\n\n')

        print('Output from convolution corr:')
        print(y_corr)

        plt.figure(figsize=(10,6))
        plt.plot(np.arange(len(y_freq)),y_freq,color='b')
        plt.plot(np.arange(len(y_corr)),y_corr,color='r')
        ax=plt.gca()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.savefig(os.path.join('','figures','compare_correlation_in_sliced_cardiogram.png'))
        plt.show()
#                       Answer-Question 3
# It seems that both the frequency-domain convolution and correlation operations result in the same output, except for the first few elements which are zero in the correlation output. 
# This is expected because convolution in the frequency domain is equivalent to correlation in the time domain, with the difference being the arrangement of the input signals.
# The convolution result is a sequence of values that represents the overlap between the two signals as one is slid over the other. 
# In this case, it appears that the convolution output gradually decreases in magnitude.

if __name__=='__main__':
    ecg_signal=ECG()
    ecg_signal.plot_signal()
    ecg_signal.statistics()
    ecg_signal.plot_sliced_and_normalized()
    ecg_signal.plot_sliced_and_correlated()
    ecg_signal.plot_sliced_convolution_freq()
