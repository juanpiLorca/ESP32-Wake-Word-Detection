import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging (1)
import core
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from model import create_model

# Set the seed value for experiment reproducibility.
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)
print(tf.config.list_physical_devices('GPU'))

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

    train_spectrogram_ds = make_spec_ds(train_ds)
    val_spectrogram_ds = make_spec_ds(val_ds)
    test_spectrogram_ds = make_spec_ds(test_ds)

    for example_spectrograms, example_spect_labels in train_spectrogram_ds.take(1):
        break

    cnt = 0
    for batch,_ in train_spectrogram_ds:
        cnt += 1
    print("Num batches:", cnt)

    input_shape = example_spectrograms.shape[1:]  # (49,40,1)
    print('Input shape:', input_shape)
    num_labels = len(label_names)

    norm_layer = tf.keras.layers.Normalization()
    # Adapt normalization layer
    norm_layer.adapt(
        train_spectrogram_ds.map(lambda spec, label: tf.cast(spec, tf.float32))
    )

    model = create_model(input_shape, num_labels, norm_layer=norm_layer)
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )

    EPOCHS = 100
    history = model.fit(
        train_spectrogram_ds,
        validation_data=val_spectrogram_ds,
        epochs=EPOCHS,
        callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
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


