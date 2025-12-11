import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging (1)
import core
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from model import make_esp32_keyword_model

# Set the seed value for experiment reproducibility.
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)
print(tf.config.list_physical_devices('GPU'))

preprocess_imp = 'custom'
sample_rate = 16000
clip_duration_ms = 1000
window_size_ms = 30.0
window_stride_ms = 20.0
feature_bin_count = 40
preprocess = 'mfcc'

BATCH_SIZE = 64

def make_spec_ds(ds):
    return (
        ds
        .unbatch()  # process one sample at a time
        .map(
            lambda audio, label: (
                core.wav_to_features(
                    sample_rate,
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
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

#%% Plotting Spectrograms
def build_mfcc_dataset(ds, model_settings):
    mfcc_list = []
    label_list = []

    for audio, label in ds.unbatch():              
        audio_np = audio.numpy().astype(np.float32)

        mfcc, _ = core.compute_audio_mfcc_features(audio_np, model_settings)
        # Expand channel dimension for CNNs → (frames, coeffs, 1)
        mfcc = np.expand_dims(mfcc, axis=-1)
        mfcc_list.append(mfcc)
        label_list.append(label.numpy())

    mfcc_array = np.array(mfcc_list, dtype=np.float32)
    ds_out = tf.data.Dataset.from_tensor_slices((mfcc_array, label_list))
    return ds_out


if __name__ == "__main__":

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

    model_settings = core.prepare_model_settings(   
	len(label_names), preprocess_imp, sample_rate, clip_duration_ms, window_size_ms,
	window_stride_ms, feature_bin_count, preprocess)    
    print(f'Model settings: {model_settings}')

    print("Building MFCC training dataset...")
    train_mfcc_ds = build_mfcc_dataset(train_ds, model_settings)
    print("Building MFCC validation dataset...")
    val_mfcc_ds = build_mfcc_dataset(val_ds, model_settings)
    print("Building MFCC test dataset...")
    test_mfcc_ds = build_mfcc_dataset(test_ds, model_settings)

    train_mfcc_ds = train_mfcc_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_mfcc_ds = val_mfcc_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    test_mfcc_ds = test_mfcc_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    for example_spectrograms, example_spect_labels in train_mfcc_ds.take(1):
        break

    cnt = 0
    for batch,_ in train_mfcc_ds:
        cnt += 1
    print("Num batches:", cnt)

    input_shape = example_spectrograms.shape[1:]  # (49,40,1)
    print('Input shape:', input_shape)
    num_labels = len(label_names)

    norm_layer = tf.keras.layers.Normalization()
    # Adapt normalization layer
    norm_layer.adapt(
        train_mfcc_ds.map(lambda spec, label: tf.cast(spec, tf.float32))
    )

    model = make_esp32_keyword_model(input_shape, num_labels)
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )

    EPOCHS = 200
    history = model.fit(
        train_mfcc_ds,
        validation_data=val_mfcc_ds,
        epochs=EPOCHS,
        callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=8, restore_best_weights=True),
    )

    metrics = history.history
    plt.figure(figsize=(16,6))
    plt.subplot(1,2,1)
    plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
    plt.legend(['loss', 'val_loss'])
    plt.ylim([0, max(plt.ylim())])
    plt.xlabel('Epoch')
    plt.ylabel('Loss [CrossEntropy]')

    plt.subplot(1,2,2)
    plt.plot(history.epoch, 100*np.array(metrics['accuracy']), 100*np.array(metrics['val_accuracy']))
    plt.legend(['accuracy', 'val_accuracy'])
    plt.ylim([0, 100])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy [%]')
    plt.show()


