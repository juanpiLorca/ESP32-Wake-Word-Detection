#%% Imports 
import core
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from IPython import display

# Set the seed value for experiment reproducibility.
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)
print(tf.config.list_physical_devices('GPU'))

#%% Load Datasets: Allocated in GPU Memory
train_ds, val_ds = tf.keras.utils.audio_dataset_from_directory(
	directory="data/speech_commands/train",
	batch_size=64,
	validation_split=0.2,
	seed=0,
	output_sequence_length=16000,
	subset='both'
)
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

for example_audio, example_labels in train_ds.take(1):
	print("Batch audio shape:", example_audio.shape)
	print("Batch tag shape:", example_labels.shape)
for example_test_audio, example_test_labels in test_ds.take(1):
	print("Test Batch audio shape:", example_test_audio.shape)
	print("Test Batch tag shape:", example_test_labels.shape)

#%% Plot audio waveforms
label_names[[3,2,1,0]]

plt.figure(figsize=(16, 10))
rows = 3
cols = 3
n = rows * cols
for i in range(n):
	plt.subplot(rows, cols, i+1)
	audio_signal = example_audio[i]
	plt.plot(audio_signal)
	plt.title(label_names[example_labels[i]])
	plt.yticks(np.arange(-1.2, 1.2, 0.2))
	plt.ylim([-1.1, 1.1])

#%% Prepare Model Settings: Embeddings Parameters for FFT
preprocess_imp = 'custom'
sample_rate = 16000
clip_duration_ms = 1000
window_size_ms = 25.0
window_stride_ms = 10.0
feature_bin_count = 40
preprocess = 'mfcc'

model_settings = core.prepare_model_settings(   
	len(label_names), preprocess_imp, sample_rate, clip_duration_ms, window_size_ms,
	window_stride_ms, feature_bin_count, preprocess)    
print(f'Model settings: {model_settings}')

for i in range(3):
	label = label_names[example_labels[i]]
	waveform = example_audio[i]
	features, spectrogram = core.wav_to_features(
			sample_rate, preprocess_imp, clip_duration_ms, window_size_ms,
			window_stride_ms, feature_bin_count, preprocess,
			input_audio=waveform, spectrogram_output=True)
	print('Label:', label)
	print('Waveform shape:', waveform.shape)
	print('Feature shape:', features.shape)
	print('Spectrogram shape:', spectrogram.shape)
	print('Audio playback')
	display.display(display.Audio(waveform[:,0], rate=16000))

#%% Plotting Spectrograms
def build_mfcc_dataset(ds, model_settings):
    mfcc_list = []
    label_list = []

    for audio, label in ds.unbatch():               # process item by item
        audio_np = audio.numpy().astype(np.float32)

        mfcc, _ = core.compute_audio_mfcc_features(audio_np, model_settings)

        # Expand channel dimension for CNNs → (frames, coeffs, 1)
        mfcc = np.expand_dims(mfcc, axis=-1)

        mfcc_list.append(mfcc)
        label_list.append(label.numpy())

    mfcc_array = np.array(mfcc_list, dtype=np.float32)

    ds_out = tf.data.Dataset.from_tensor_slices((mfcc_array, label_list))
    return ds_out


def make_spec_ds(ds):
	return (
		ds
		.unbatch()  # process one sample at a time
		.map(
			lambda audio, label: (
				core.wav_to_features(
						sample_rate,
						preprocess_imp,
						clip_duration_ms,
						window_size_ms,
						window_stride_ms,
						feature_bin_count,
						preprocess,         
						input_audio=audio
				),
				label
			),
			num_parallel_calls=tf.data.AUTOTUNE
		)
		.batch(64)
		.prefetch(tf.data.AUTOTUNE)
	)

if preprocess_imp == 'tensorflow':
	train_spectrogram_ds = make_spec_ds(train_ds)
	val_spectrogram_ds = make_spec_ds(val_ds)
	test_spectrogram_ds = make_spec_ds(test_ds)
else:
	print("Building MFCC training dataset...")
	train_mfcc_ds = build_mfcc_dataset(train_ds, model_settings)
	print("Building MFCC validation dataset...")
	val_mfcc_ds = build_mfcc_dataset(val_ds, model_settings)
	print("Building MFCC test dataset...")
	test_mfcc_ds = build_mfcc_dataset(test_ds, model_settings)

	batch_size = 64
	train_mfcc_ds = train_mfcc_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
	val_mfcc_ds = val_mfcc_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
	test_mfcc_ds = test_mfcc_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

#%% Plot example spectrograms
def plot_spectrogram(spectrogram, ax, preprocess=preprocess):
	if len(spectrogram.shape) > 2:
		assert len(spectrogram.shape) == 3
		spectrogram = np.squeeze(spectrogram, axis=-1)
	# Convert the frequencies to log scale and transpose, so that the time is
	# represented on the x-axis (columns).
	# Add an epsilon to avoid taking a log of zero.
	log_spec = spectrogram.T
	# else:
	# 	log_spec = np.log(spectrogram.T + np.finfo(float).eps)
	height = log_spec.shape[0]
	width = log_spec.shape[1]
	X = np.linspace(0, width, num=width, dtype=int)
	Y = range(height)
	ax.pcolormesh(X, Y, log_spec)

print(train_mfcc_ds)
if preprocess_imp == 'tensorflow':
	for example_spectrograms, example_spect_labels in train_spectrogram_ds.take(1):
		break
else: 
	for example_spectrograms, example_spect_labels in train_mfcc_ds.take(1):
		print("Example MFCC batch shape:", example_spectrograms.shape)
		print("Example MFCC labels shape:", example_spect_labels.shape)
		break

rows = 4
cols = 4
n = rows*cols
fig, axes = plt.subplots(rows, cols, figsize=(16, 9))

for i in range(n):
    r = i // cols
    c = i % cols
    ax = axes[r][c]
    plot_spectrogram(example_spectrograms[i].numpy(), ax)
    ax.set_title(label_names[example_spect_labels[i].numpy()])
plt.show()




