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

data3 = pm.datasets.load_wineind()
train, test = data3[:150], data[150:]
lenSeq = 10000
subSamp = 40
f0Samples = 400

data.shape
data = data[:lenSeq]

with StepwiseContext(max_steps=2):
  pipe = pipeline.Pipeline([
      ("fourier", ppc.FourierFeaturizer(m=f0Samples)),
      ("arima", arima.AutoARIMA(stepwise=True, maxiter=20, with_intercept = False, start_p=5, start_q=4,  max_p= 6, max_q= 6,  trace=1, error_action="ignore",
                              seasonal=False,  # because we use Fourier
                              suppress_warnings=True))
  ])

  pipe.fit(data)
  yhat = pipe.predict(n_periods=10000)

data1 = np.array(data)
yhat1 = np.array(yhat)
output = np.concatenate((data1,yhat1))
write('output_string.wav',44100,output)

thisfig = plt.figure(figsize=(12,8))
plt.plot(np.arange(1,lenSeq+1), data, label='Real Sequence', color='blue')
plt.plot(np.arange(lenSeq,lenSeq+len(yhat)), yhat, label='Forecast-', color='green')
plt.legend(['data','predict'], loc=3)
plt.xlabel('Length')
plt.ylabel('Amplitude')
plt.show()
thisfig.savefig("Pred05.pdf", bbox_inches='tight')
plt.close(); print('\n')
