from scipy.io import wavfile
from scipy.io.wavfile import write
fs, data = wavfile.read('1000kV输电线噪声.wav')
import pmdarima as pm
from pmdarima import pipeline
from pmdarima import model_selection
from pmdarima import preprocessing as ppc
from pmdarima import arima
from pmdarima.arima import StepwiseContext
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from scipy import signal

data3 = pm.datasets.load_wineind()
train, test = data3[:150], data[150:]
lenSeq = 10000
downsampling_rate = 300
subSamp = 40
f0SamplesSS = 10
f0Samples = 400
stdLength = 1000
predLength = 10000

#Calculate the Energy of the audio signal and down-sample it.
data = data[:lenSeq]
data2 = pd.Series(data)
mstd1 = data2.rolling(stdLength, min_periods=0, center=True).std()
mstd2 = np.array(mstd1)
mstd3 = np.nan_to_num(mstd2)
mstd_downsample = mstd3[::downsampling_rate]

#Predict Energy (From Downsampled Signal)
stdArimaModel = pm.auto_arima(mstd_downsample, start_p=1, start_q=1, max_p=5, max_q=5, m=20,
                             start_P=0, start_Q=0, seasonal=True, start_d=1, max_d=5, D=1, trace=True,
                             error_action='ignore',  # don't want to know if an order does not work
                             suppress_warnings=True)  # set to stepwise
stdArimaModel.fit(mstd_downsample)
yhatEnergy = stdArimaModel.predict(n_periods=int(predLength/downsampling_rate))
yhatEnergy = np.clip(yhatEnergy, 0, None)
yhatEnergy = signal.resample(yhatEnergy, predLength)  # up-sample the predicted energy
yhatEnergy = np.clip(yhatEnergy, 0, None)

#Predict Signal
with StepwiseContext(max_steps=2):
  pipe = pipeline.Pipeline([
      ("fourier", ppc.FourierFeaturizer(m=f0Samples)),
      ("arima", arima.AutoARIMA(stepwise=True, maxiter=20, with_intercept = False, start_p=5, start_q=4,  max_p= 6, max_q= 6,  trace=1, error_action="ignore",
                             seasonal=False,  # because we use Fourier
                              suppress_warnings=True))
  ])

pipe.fit(data)
yhat = pipe.predict(n_periods=predLength)


#Normalise Signal with predicted energy
non_Norm_yhat = pd.Series(yhat)
non_Norm_yhat_energy = non_Norm_yhat.rolling(stdLength, min_periods=1, center=True).std()
non_Norm_yhat_energy = np.array(non_Norm_yhat_energy)
non_Norm_yhat_energy = np.nan_to_num(non_Norm_yhat_energy)
normalising_signal = yhatEnergy/non_Norm_yhat_energy

fullEnergy1 = np.concatenate((mstd3,non_Norm_yhat_energy))
fullEnergy2 = np.concatenate((mstd3,yhatEnergy))
norm_yhat_out = yhat*normalising_signal

non_Norm_Out = np.concatenate((data,yhat))
full_signal_out = np.concatenate((data,norm_yhat_out))
newLen = len(full_signal_out)



thisfig = plt.figure(figsize=(12,8))
plt.plot(np.arange(1,newLen+1), fullEnergy1, label='Real Sequence', color='red')
plt.plot(np.arange(1,newLen+1), fullEnergy2, label='Real Sequence', color='pink')
plt.plot(np.arange(1,newLen+1), non_Norm_Out , label='Real Sequence', color='green')
plt.plot(np.arange(1,newLen+1), full_signal_out , label='Real Sequence', color='blue')
plt.legend(['full energy1','full energy2','non norm output','full signal out'],
           loc=3)
plt.xlabel('Length')
plt.ylabel('Amplitude')
#Output the audio of the full output signal: full_signal_out
write('1000kV输电线噪声预测.wav', 44100, full_signal_out)

plt.show()
thisfig.savefig("Pred07.pdf", bbox_inches='tight')
plt.close()
print('\n')
