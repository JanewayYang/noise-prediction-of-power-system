from scipy.io import wavfile
from scipy.io.wavfile import write
fs, data = wavfile.read('1000kV输电线噪声.wav')
import numpy as np
import pmdarima as pm
import matplotlib
import matplotlib.pyplot as plt

data3 = pm.datasets.load_wineind()
train, test = data[:150], data[150:]
lenSeq = 10000
subSamp = 40

data = data[:lenSeq]

thisfig = plt.figure(figsize=(12,8))
ln1 = plt.plot(np.arange(1,lenSeq+1), data, label='Real Sequence', color='blue')
plt.legend(['power distribution'], loc=3)
plt.xlabel('Length')
plt.ylabel('Amplitude')
plt.show()
thisfig.savefig("Pred01.pdf", bbox_inches='tight')
plt.close()
print('\n')