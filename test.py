import core
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display

# Set the seed value for experiment reproducibility.
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

print(tf.config.list_physical_devices('GPU'))


train_ds, val_ds = tf.keras.utils.audio_dataset_from_directory(
    directory="data/speech_commands/train",
    batch_size=64,
    validation_split=0.2,
    seed=0,
    output_sequence_length=16000,
    subset='both')

test_ds = tf.keras.utils.audio_dataset_from_directory(
    directory="data/speech_commands/test",
    batch_size=64,
    validation_split=0.0,
    seed=0,    
    output_sequence_length=16000
)


label_names = np.array(train_ds.class_names)
print()
print("label names:", label_names)

for example_audio, _ in train_ds.take(1): 
    print("Audio shape:", example_audio.shape)

for exmaple_test_audio, _ in test_ds.take(1):
    print("Test audio shape:", exmaple_test_audio.shape)

sample_rate = 16000
clip_duration_ms = 1000
window_size_ms = 30.0
window_stride_ms = 20.0
feature_bin_count = 40
preprocess = 'average'

model_settings = core.prepare_model_settings(   
    len(label_names), sample_rate, clip_duration_ms, window_size_ms,
    window_stride_ms, feature_bin_count, preprocess)

print(f'Model settings: {model_settings}')

waveform = example_audio[63]
features, spectrogram = core.wav_to_features(
    sample_rate, clip_duration_ms, window_size_ms,
    window_stride_ms, feature_bin_count, preprocess,
    input_audio=waveform)  
print(f'Audio sample - feature shape: {features.shape}')
print(f'Audio sample - spectrogram shape: {spectrogram.shape}')

spectrogram = tf.nn.pool(
    input=tf.expand_dims(spectrogram, -1),
    window_shape=[1, model_settings['average_window_width']],
    strides=[1, model_settings['average_window_width']],
    pooling_type='AVG',
    padding='SAME'
)

print(f'Pooled spectrogram shape: {spectrogram.shape}')

# features is shape (49, 40)
plt.figure(figsize=(8, 5))
plt.imshow(
    features[:,:,0], 
    aspect='auto', 
    origin='lower', 
    cmap='magma'
)
plt.colorbar(label="Magnitude")
plt.xlabel("Time frames")
plt.ylabel("Frequency bins")
plt.title("STFT Spectrogram")
plt.tight_layout()
plt.show()