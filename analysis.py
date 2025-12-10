#%% Imports 
import core
import librosa.display
import numpy as np 
import librosa
import librosa.display
import matplotlib.pyplot as plt

import tensorflow as tf
from IPython import display

#%% Compute spectrograms for "up" and "down" using model_settings

# Load both files using the same sample rate as in model_settings
file_up = "examples/up/2aa787cf_nohash_0.wav"
file_down = "examples/down/2d82a556_nohash_0.wav"

y_up, sr_up = librosa.load(file_up, sr=None)
y_down, sr_down = librosa.load(file_down, sr=None)
num_samples = len(y_up)
time = np.arange(num_samples) / sr_up
# Plot the waveforms
scale = np.iinfo(np.int16).max
fig,(ax1, ax2) = plt.subplots(2,1, figsize=(10,6), sharex=True)
ax1.plot(time, (y_up * scale).astype(np.int16))
ax2.plot(time, (y_down * scale).astype(np.int16))
ax1.set_title("Waveform of 'UP' command")
ax2.set_title("Waveform of 'DOWN' command")
ax2.set_xlabel("Time (s)")
plt.tight_layout()
plt.show()

librosa.display.waveshow(y_up, sr=sr_up)
librosa.display.waveshow(y_down, sr=sr_down)
plt.show()

#%%
display.Audio(y_up, rate=sr_up)
#%%
display.Audio(y_down, rate=sr_down)

#%%
sample_rate = 16000
clip_duration_ms = 1000
window_size_ms = 25.0
window_stride_ms = 10.0
feature_bin_count = 40
preprocess = 'mfcc'
model_settings = core.prepare_model_settings(
    label_count=0, 
    sample_rate=sample_rate,
    clip_duration_ms=clip_duration_ms,
    window_size_ms=window_size_ms,
    window_stride_ms=window_stride_ms,
    feature_bin_count=feature_bin_count,
    preprocess=preprocess
)

for key in model_settings.keys():
    print(f"{key}: {model_settings[key]}")

# Convenience variables from model_settings
hop = model_settings["window_stride_samples"]
win = model_settings["window_size_samples"]

features = core.wav_to_features(
    sample_rate=sr_up,
    clip_duration_ms=clip_duration_ms,
    window_size_ms=window_size_ms,
    window_stride_ms=window_stride_ms,
    feature_bin_count=feature_bin_count,
    preprocess=preprocess,
    input_audio=tf.convert_to_tensor(y_up)[:, tf.newaxis]
)
print("Features shape:", features.shape)

# STFT + magnitude spectrograms
S_up, phase_up = librosa.magphase(
    librosa.stft(
        y=y_up,
        n_fft=win,
        hop_length=hop,
        win_length=win,
        window='hamm'
    )
)

S_down, phase_down = librosa.magphase(
    librosa.stft(
        y=y_down,
        n_fft=win,
        hop_length=hop,
        win_length=win,
        window='hamm'
    )
)

S_up_db = librosa.amplitude_to_db(S_up, ref=np.max)
S_down_db = librosa.amplitude_to_db(S_down, ref=np.max)

print("UP spectrogram shape:", S_up_db.shape)
print("DOWN spectrogram shape:", S_down_db.shape)

#%%
def plot_spectrogram(spectrogram, ax, preprocess=preprocess):
	if len(spectrogram.shape) > 2:
		assert len(spectrogram.shape) == 3
		spectrogram = np.squeeze(spectrogram, axis=-1)
	# Convert the frequencies to log scale and transpose, so that the time is
	# represented on the x-axis (columns).
	# Add an epsilon to avoid taking a log of zero.
	if preprocess == 'mfcc':
		log_spec = spectrogram.T
	else:
		log_spec = np.log(spectrogram.T + np.finfo(float).eps)
	height = log_spec.shape[0]
	width = log_spec.shape[1]
	X = np.linspace(0, width, num=width, dtype=int)
	Y = range(height)
	ax.pcolormesh(X, Y, log_spec)

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
plot_spectrogram(features.numpy(), ax,preprocess=preprocess)
plt.show()

#%% Plot both spectrograms one under the other
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# --- UP ---
img1 = librosa.display.specshow(
    S_up_db,
    sr=sr_up,
    hop_length=hop,
    y_axis='log',
    x_axis='time',
    ax=ax1
)
ax1.set_title("UP command - Spectrogram")
fig.colorbar(img1, ax=ax1, format="%+2.0f dB")

# --- DOWN ---
img2 = librosa.display.specshow(
    S_down_db,
    sr=sr_down,
    hop_length=hop,
    y_axis='log',
    x_axis='time',
    ax=ax2
)
ax2.set_title("DOWN command - Spectrogram")
fig.colorbar(img2, ax=ax2, format="%+2.0f dB")

plt.tight_layout()
plt.show()

#%% Compute MFCCs for both commands
n_mfcc = 12#model_settings["fingerprint_width"]

mfcc_up = librosa.feature.mfcc(
    y=y_up,
    sr=sr_up,
    n_mfcc=n_mfcc,
    n_fft=win,
    hop_length=hop,
    win_length=win,
    window="hann"
)

mfcc_down = librosa.feature.mfcc(
    y=y_down,
    sr=sr_down,
    n_mfcc=n_mfcc,
    n_fft=win,
    hop_length=hop,
    win_length=win,
    window="hann"
)

print("MFCC UP shape:", mfcc_up.shape)
print("MFCC DOWN shape:", mfcc_down.shape)

#%% Plot both MFCCs one under the other

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

img1 = librosa.display.specshow(
    mfcc_up,
    sr=sr_up,
    hop_length=hop,
    x_axis='time',
    ax=ax1
)
ax1.set_title("MFCC - UP Command")
fig.colorbar(img1, ax=ax1)

img2 = librosa.display.specshow(
    mfcc_down,
    sr=sr_down,
    hop_length=hop,
    x_axis='time',
    ax=ax2
)
ax2.set_title("MFCC - DOWN Command")
fig.colorbar(img2, ax=ax2)

plt.tight_layout()
plt.show()

