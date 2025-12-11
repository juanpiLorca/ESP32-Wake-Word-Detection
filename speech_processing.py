#%% Imports
import numpy as np 
import scipy.io.wavfile as wav
from scipy.fftpack import dct
import matplotlib.pyplot as plt

import librosa
import librosa.display

#%% Load audio files 

file_up = "data/speech_commands/test/up/2aa787cf_nohash_0.wav"
file_down = "data/speech_commands/test/down/2d82a556_nohash_0.wav"

sr, xup = wav.read(file_up)
sr, xdown = wav.read(file_down)

time = np.linspace(0, len(xup) / sr, num=len(xup))

plt.plot(time, xup)
plt.plot(time, xdown)
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.title("Audio Signals")
plt.legend(["UP", "DOWN"])
plt.show()

#%% Pre-emphasis filter
def pre_emphasis(signal, pre_emphasis_coeff=0.97):
    emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis_coeff * signal[:-1])
    return emphasized_signal

xup_emphasized = pre_emphasis(xup)
xdown_emphasized = pre_emphasis(xdown)

plt.plot(time, xup_emphasized)
plt.plot(time, xdown_emphasized)
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.title("Pre-emphasized Audio Signals")
plt.legend(["UP", "DOWN"])
plt.show()

#%% Framing
frame_size_ms = 25
frame_stride_ms = frame_size_ms * 0.5
frame_size_length = int(round(sr * frame_size_ms / 1000))
frame_stride_length = int(round(sr * frame_stride_ms / 1000))
print(f"Frame size: {frame_size_length}, Frame stride: {frame_stride_length}")

xup_emphasized_length = len(xup_emphasized)
xdown_emphasized_length = len(xdown_emphasized)

num_frames = int(np.ceil(float(np.abs(xup_emphasized_length - frame_size_length)) / frame_stride_length))  
pad_signal_length = num_frames * frame_stride_length + frame_size_length

z = np.zeros(shape=((pad_signal_length - xup_emphasized_length),))
pad_up_signal = np.append(xup_emphasized, z) 
pad_down_signal = np.append(xdown_emphasized, z) 

indices = np.tile(np.arange(0, frame_size_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_stride_length, frame_stride_length), (frame_size_length, 1)).T
up_frames = pad_up_signal[indices.astype(np.int32, copy=False)]
down_frames = pad_down_signal[indices.astype(np.int32, copy=False)]
print(f"Number of frames: {num_frames}, Frame shape: {up_frames.shape}")

#%% Windowing
hamming_window = np.hamming(frame_size_length)
up_frames_windowed = up_frames * hamming_window
down_frames_windowed = down_frames * hamming_window

#%% FFT and Power Spectrum
nfft = 512 
up_mag_frames = np.absolute(np.fft.rfft(up_frames_windowed, n=nfft))
down_mag_frames = np.absolute(np.fft.rfft(down_frames_windowed, n=nfft))
up_pow_frames = ((1.0 / nfft) * (up_mag_frames ** 2))
down_pow_frames = ((1.0 / nfft) * (down_mag_frames ** 2))
print(f"Power spectrum shape: {up_pow_frames.shape}")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
imgUp = librosa.display.specshow(
    librosa.power_to_db(up_pow_frames.T, ref=np.max),
    sr=sr, hop_length=frame_stride_length, x_axis='time', y_axis='linear', ax=ax1)
ax1.set_title("UP command - Power Spectrum")

imgDown = librosa.display.specshow(
    librosa.power_to_db(down_pow_frames.T, ref=np.max),
    sr=sr, hop_length=frame_stride_length, x_axis='time', y_axis='linear', ax=ax2)
ax2.set_title("DOWN command - Power Spectrum")

plt.colorbar(imgUp, ax=ax1, format="%+2.0f dB")
plt.colorbar(imgDown, ax=ax2, format="%+2.0f dB")

plt.tight_layout()
plt.show()

#%% Mel Filter Banks
nfilt = 40
low_freq_mel = 0
high_freq_mel = (2595 * np.log10(1 + (sr / 2) / 700))  # Convert Hz to Mel
mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
bin = np.floor((nfft + 1) * hz_points / sr)

def gen_filter_banks(nfilt, nfft, bin):
    fbank = np.zeros((nfilt, int(np.floor(nfft / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    return fbank

fbank = gen_filter_banks(nfilt, nfft, bin)
up_filter_banks = np.dot(up_pow_frames, fbank.T)
up_filter_banks = np.where(up_filter_banks == 0, np.finfo(float).eps, up_filter_banks)  # Numerical Stability
up_filter_banks = 20 * np.log10(up_filter_banks)
down_filter_banks = np.dot(down_pow_frames, fbank.T)
down_filter_banks = np.where(down_filter_banks == 0, np.finfo(float).eps, down_filter_banks)  
down_filter_banks = 20 * np.log10(down_filter_banks)  
print(f"Filter banks shape: {up_filter_banks.shape}")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
imgUpFB = librosa.display.specshow(
    up_filter_banks.T, sr=sr, hop_length=frame_stride_length, x_axis='time', y_axis='mel', ax=ax1)
ax1.set_title("UP command - Mel Filter Banks")

imgDownFB = librosa.display.specshow(
    down_filter_banks.T, sr=sr, hop_length=frame_stride_length, x_axis='time', y_axis='mel', ax=ax2)
ax2.set_title("DOWN command - Mel Filter Banks")
plt.colorbar(imgUpFB, ax=ax1, format="%+2.0f dB")
plt.colorbar(imgDownFB, ax=ax2, format="%+2.0f dB")

plt.tight_layout()
plt.show()

#%% Mean normalization
up_filter_banks -= (np.mean(up_filter_banks, axis=0) + 1e-8)
down_filter_banks -= (np.mean(down_filter_banks, axis=0) + 1e-8)
print("Mean normalization done.")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
imgUpFBNorm = librosa.display.specshow(
    up_filter_banks.T, sr=sr, hop_length=frame_stride_length, x_axis='time', y_axis='mel', ax=ax1)
ax1.set_title("UP command - Normalized Mel Filter Banks")

imgDownFBNorm = librosa.display.specshow(
    down_filter_banks.T, sr=sr, hop_length=frame_stride_length, x_axis='time', y_axis='mel', ax=ax2)
ax2.set_title("DOWN command - Normalized Mel Filter Banks")
plt.colorbar(imgUpFBNorm, ax=ax1, format="%+2.0f dB")
plt.colorbar(imgDownFBNorm, ax=ax2, format="%+2.0f dB")

plt.tight_layout()
plt.show()

#%% MFCCs
num_ceps = 12
up_mfcc = dct(up_filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)]
down_mfcc = dct(down_filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)]
print(f"MFCC shape: {up_mfcc.shape}")

cep_lifter = 22
(nframes, ncoeff) = up_mfcc.shape
n = np.arange(ncoeff)
lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
up_mfcc *= lift
down_mfcc *= lift

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
imgUpMFCC = librosa.display.specshow(
    up_mfcc.T, sr=sr, hop_length=frame_stride_length, x_axis='time', y_axis='mel', ax=ax1)
ax1.set_title("UP command - MFCCs")

imgDownMFCC = librosa.display.specshow(
    down_mfcc.T, sr=sr, hop_length=frame_stride_length, x_axis='time', y_axis='mel', ax=ax2)
ax2.set_title("DOWN command - MFCCs")
plt.colorbar(imgUpMFCC, ax=ax1)
plt.colorbar(imgDownMFCC, ax=ax2)

plt.tight_layout()
plt.show()

#%% Normalized MFCCs
# Per frame mean normalization
up_mfcc_norm = (up_mfcc - up_mfcc.mean(axis=0) + 1e-8) / (up_mfcc.std(axis=0) + 1e-8)
down_mfcc_norm = (down_mfcc - down_mfcc.mean(axis=0) + 1e-8) / (down_mfcc.std(axis=0) + 1e-6)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
imgUpMFCCNorm = librosa.display.specshow(
    up_mfcc_norm.T, sr=sr, hop_length=frame_stride_length, x_axis='time', y_axis='mel', ax=ax1)
ax1.set_title("UP command - Normalized MFCCs")

imgDownMFCCNorm = librosa.display.specshow(
    down_mfcc_norm.T, sr=sr, hop_length=frame_stride_length, x_axis='time', y_axis='mel', ax=ax2)
ax2.set_title("DOWN command - Normalized MFCCs")
plt.colorbar(imgUpMFCCNorm, ax=ax1)
plt.colorbar(imgDownMFCCNorm, ax=ax2)

plt.tight_layout()
plt.show()