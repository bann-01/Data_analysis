import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter
from scipy import signal
from scipy.signal import lfilter

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_lowpass(cutoff, fs, order=9):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=9):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_highpass(cutoff, fs, order=3):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=3):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

############### Examples ###################

Fs = 2000           # Maximum frecs.
Ts = 1 / Fs         # Time bins.
te = 1.0
t = np.arange(0.0, te, Ts)

# Signal x (20Hz) + Signal y (50Hz)
x = np.cos(2 * np.pi * 20 * t)
y = np.cos(2 * np.pi * 50 * t)
# Signal z
z = x + y

N = len(z)

k = np.arange(N)
T = N / Fs
freq = k / T
freq = freq[range(int(N/2))]


# HPF
cutoff = 50.
hpf = butter_highpass_filter(z, cutoff, Fs)

# 1. 원 신호
plt.plot(t, z, 'y', label='origin')

# 2. 필터 적용된 Plot
plt.plot(t, hpf, 'b', label='filtered data')
plt.legend()
plt.show() 

# 3. 필터 적용된 FFT Plot
yfft = np.fft.fft(hpf)
yf = yfft / N
yf = yf[range(int(N/2))]

plt.plot(freq, abs(yf), 'b')
plt.title("HBF")
plt.xlim(0, Fs / 20)
plt.show()