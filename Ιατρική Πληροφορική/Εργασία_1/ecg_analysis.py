import os,numpy as np
import matplotlib.pyplot as plt
import statistics
from scipy.stats import skew,kurtosis
from tabulate import tabulate

class Signal:
    def __init__(self) -> None:
        pass
    
    def sread(self,path_to_signal:str):
        self.ecg_signal=np.loadtxt(path_to_signal)
        self.random_noisy_ecg=0.5*np.random.randn(len(self.ecg_signal))
        self.noisy_ecg_signal=self.ecg_signal+self.random_noisy_ecg

        # Numpy correlation calculation
        self.corr_ecg_signal_with_noise=np.correlate(self.ecg_signal,self.noisy_ecg_signal,mode='same')
        self.corr_random_signal=np.correlate(self.ecg_signal,self.random_noisy_ecg,mode='same')

    def statistics(self):
        q1,q3,_=statistics.quantiles(data=self.ecg_signal,n=4)
        rows=[
            ['Samples',self.ecg_signal.shape[0]],
            ['Mean',statistics.mean(self.ecg_signal)],
            ['Median',statistics.median(self.ecg_signal)],
            ['Std',statistics.stdev(self.ecg_signal)],
            ['Iqr',q3-q1],
            ['Skewness',skew(self.ecg_signal)],
            ['Kurtosis',kurtosis(self.ecg_signal)]
        ]
        print(tabulate(tabular_data=rows,headers=['Statistic Meter','Value'],tablefmt='fancy_grid',floatfmt='.3f'))
        with open(os.path.join('stats.tex'),'w') as writer:
            writer.write(tabulate(tabular_data=rows,headers=['Statistic Meter','Value'],tablefmt='latex',floatfmt='.3f'))

    def plotall(self):                   
        fig,axs=plt.subplots(nrows=3,ncols=2,figsize=(8,8))

        for i in range(2):
            axs[0,i%2].set_title('ECG')
            axs[0,i%2].plot(np.arange(len(self.ecg_signal)),self.ecg_signal,color='blue',linewidth=2)
            axs[0,i%2].spines['top'].set_visible(False)
            axs[0,i%2].spines['right'].set_visible(False)
        
        axs[1,0].set_title('ECG + Noise')
        axs[1,0].plot(np.arange(len(self.noisy_ecg_signal)),self.noisy_ecg_signal,color='blue',linewidth=2)
        axs[1,0].spines['top'].set_visible(False)
        axs[1,0].spines['right'].set_visible(False)

        axs[1,1].set_title('Random Noise')
        axs[1,1].plot(np.arange(len(self.random_noisy_ecg)),self.random_noisy_ecg,color='blue',linewidth=2)
        axs[1,1].spines['top'].set_visible(False)
        axs[1,1].spines['right'].set_visible(False)

        axs[2,0].set_title('Correletion of ECG + Noise')
        axs[2,0].plot(np.arange(len(self.corr_ecg_signal_with_noise)),self.corr_ecg_signal_with_noise,color='blue',linewidth=2)
        axs[2,0].spines['top'].set_visible(False)
        axs[2,0].spines['right'].set_visible(False)

        axs[2,1].set_title('Corretion Ecg with Random Noise')
        axs[2,1].plot(np.arange(len(self.corr_random_signal)),self.corr_random_signal,color='blue',linewidth=2)
        axs[2,1].spines['top'].set_visible(False)
        axs[2,1].spines['right'].set_visible(False)

        fig.tight_layout()
        fig.savefig(fname=os.path.join('','ecg_askisi1.png'),dpi=300)
        plt.show()

class Signal2:
    def __init__(self) -> None:
        pass
    
    def sread(self,path_to_signal:str):
        self.ecg_signal=np.loadtxt(path_to_signal)
        f1=17
        f2=10

        self.sine1 = np.sin(2*np.pi*f1*np.arange(len(self.ecg_signal))*(1/len(self.ecg_signal)))
        self.sine2 = np.sin(2*np.pi*f2*np.arange(len(self.ecg_signal))*(1/len(self.ecg_signal)))   

        self.random_noisy_ecg=0.5*np.random.randn(len(self.ecg_signal))
        self.noisy_ecg_signal=self.ecg_signal+self.random_noisy_ecg

        # Numpy correlation calculation
        self.corr_ecg_signal_sine1=np.correlate(self.ecg_signal,self.sine1,mode='same')
        self.corr_ecg_signal_sine2=np.correlate(self.ecg_signal,self.sine2,mode='same')

    def statistics(self):
        q1,q3,_=statistics.quantiles(data=self.ecg_signal,n=4)
        rows=[
            ['Samples',self.ecg_signal.shape[0]],
            ['Mean',statistics.mean(self.ecg_signal)],
            ['Median',statistics.median(self.ecg_signal)],
            ['Std',statistics.stdev(self.ecg_signal)],
            ['Iqr',q3-q1],
            ['Skewness',skew(self.ecg_signal)],
            ['Kurtosis',kurtosis(self.ecg_signal)]
        ]
        print(tabulate(tabular_data=rows,headers=['Statistic Meter','Value'],tablefmt='fancy_grid',floatfmt='.3f'))
        with open(os.path.join('stats.tex'),'w') as writer:
            writer.write(tabulate(tabular_data=rows,headers=['Statistic Meter','Value'],tablefmt='latex',floatfmt='.3f'))


    def plotall(self):                   
        fig,axs=plt.subplots(nrows=3,ncols=2,figsize=(8,8))

        for i in range(2):
            axs[0,i%2].set_title('ECG')
            axs[0,i%2].plot(np.arange(len(self.ecg_signal)),self.ecg_signal,color='blue',linewidth=2)
            axs[0,i%2].spines['top'].set_visible(False)
            axs[0,i%2].spines['right'].set_visible(False)
        
        axs[1,0].set_title('Sin Wave, f=17')
        axs[1,0].plot(np.arange(len(self.sine1)),self.sine1,color='blue',linewidth=2)
        axs[1,0].spines['top'].set_visible(False)
        axs[1,0].spines['right'].set_visible(False)

        axs[1,1].set_title('Sin Wave, f=10')
        axs[1,1].plot(np.arange(len(self.sine2)),self.sine2,color='blue',linewidth=2)
        axs[1,1].spines['top'].set_visible(False)
        axs[1,1].spines['right'].set_visible(False)

        axs[2,0].set_title('Correletion with Sin, f=17')
        axs[2,0].plot(np.arange(len(self.corr_ecg_signal_sine1)),self.corr_ecg_signal_sine1,color='blue',linewidth=2)
        axs[2,0].spines['top'].set_visible(False)
        axs[2,0].spines['right'].set_visible(False)

        axs[2,1].set_title('Correletion with Sin, f=10')
        axs[2,1].plot(np.arange(len(self.corr_ecg_signal_sine2)),self.corr_ecg_signal_sine2,color='blue',linewidth=2)
        axs[2,1].spines['top'].set_visible(False)
        axs[2,1].spines['right'].set_visible(False)

        fig.tight_layout()
        fig.savefig(fname=os.path.join('','ecg_askisi2.png'),dpi=300)
        plt.show()

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser("Argument Parser")
    parser.add_argument("--scenario",required=True,help="Select scenario(Available options(1|2))")
    args=parser.parse_args()
    
    if int(args.scenario)==1:
        signal_handler=Signal()
        signal_handler.sread(os.path.join('','ecg.txt'))
        signal_handler.statistics()
        signal_handler.plotall()
    elif int(args.scenario)==2:
        signal_handler2=Signal2()
        signal_handler2.sread(os.path.join('','ecg.txt'))
        signal_handler2.statistics()
        signal_handler2.plotall()
    else:
        raise ValueError("Not available option")