import pandas as pd,os,numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.stats import skew

class EEG:
    path_to_datasets=os.path.join('..','files')
    def __init__(self):
        self.data=list()
        self.classes=list()
        for sample_file in [filename for filename in os.listdir(EEG.path_to_datasets) if filename.endswith('txt')]:
            self.data.append(np.loadtxt(os.path.join(EEG.path_to_datasets,sample_file)))
            class_name=sample_file.removesuffix('.txt')
            self.classes.append(class_name[0])
        
        print(f'\n{len(self.data)=}\t{len(self.classes)=}')
        for idx,class_name in enumerate(self.classes):
            print(f'Data{idx}: {class_name}')
    
    def samples(self):
        return len(self.data)

    def wavelet_decomposition(self,signal_data,num_levels=3):
        coefficients=list()
        approximations=list()
        approx,detail=pywt.dwt(signal_data,wavelet='db4')
        approximations.append(approx)
        coefficients.append(detail)

        for _ in range(num_levels - 1):
            approx, detail = pywt.dwt(approx, wavelet='db4')
            approximations.append(approx)
            coefficients.append(detail)

        coefficients.reverse()
        approximations.reverse()
        return approximations, coefficients

    def approximate_entropy(self,signal_data,m,r,s):
        N = len(signal_data)  # Length of the data
        phi = np.zeros(N - m + 1)  # Initialize the phi array

        for i in range(N - m + 1):
            temp1 = signal_data[i:i+m]  # Subsequence of length m from the data

            distances = []
            for j in range(N - m*s + 1):
                if i != j:
                    temp2 = signal_data[j:j+m*s:s]  # Subsequence of length m*s with step size s from the data
                    distances.append(np.linalg.norm(temp1-temp2))

            # Count the number of distances that are less than the threshold r
            num_similar = sum([1 for d in distances if d <= r])

            # Calculate the probability
            phi[i] = num_similar / (N - m*s + 1)

        # Calculate the approximate entropy
        return -np.log(np.mean(phi))

    def compute_apen(self,coefficients,approximations,num_levels,m,r,s):
        epoch_apen_values=list()
        for level in range(num_levels):
            epoch_apen_values.append(self.approximate_entropy(coefficients[level],m,r,s))
        for level in range(num_levels):    
            epoch_apen_values.append(self.approximate_entropy(approximations[level],m,r,s))
        
        return epoch_apen_values

    def window_split(self,signal_idx):
        num_levels=3
        sampling_frequency=173.61
        time_duration=5

        epoch_length=int(time_duration * sampling_frequency)
        num_epochs=len(self.data[signal_idx])//epoch_length
        
        m,r,s=2,0.15*np.std(self.data[signal_idx]),1

        features=list()
        for epoch in range(num_epochs):
            start=epoch*epoch_length
            end=start+epoch_length

            epoch_data=self.data[signal_idx][start:end]
            approximations,coefficients=self.wavelet_decomposition(epoch_data,num_levels=num_levels)
            features.append(self.compute_apen(coefficients=coefficients,approximations=approximations,num_levels=num_levels,m=m,r=r,s=s))
            print(f'{self.statistical_features(coefficients)=}')

        if end<len(self.data[signal_idx]):
            epoch_data=self.data[signal_idx][end:len(self.data[signal_idx])]
            approximations,coefficients=self.wavelet_decomposition(epoch_data,num_levels=num_levels)
            features.append(self.compute_apen(coefficients=coefficients,approximations=approximations,num_levels=num_levels,m=m,r=r,s=s))
            print(f'{self.statistical_features(coefficients)=}')
        return pd.DataFrame(data=features,columns=['A1','A2','A3','D1','D2','D3'])

    def statistical_features(self,coefficients):
        stat_features=list()

        for coef in coefficients:
            mav=np.mean(np.abs(coef))
            avp=np.mean(np.abs(coef**2))
            variance=np.var(coef)
            sd=np.std(coef)
            mean=np.mean(coef)
            coef_skewness=skew(coef)
            stat_features.append([mav,avp,variance,sd,mean,coef_skewness])

        return stat_features


    def plot_signal(self,signal,plottitle):
        plt.figure(figsize=(10,2))
        plt.plot(np.arange(len(signal)),signal,marker='o',color='r')
        plt.title(plottitle)
        plt.xlabel('Sample')
        plt.ylabel('mV')
        ax=plt.gca()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.show()

if __name__=='__main__':
    eeg=EEG()
    data=pd.DataFrame()
    data=pd.concat(objs=[eeg.window_split(sample_idx) for sample_idx in range(eeg.samples())])
    print(data)
