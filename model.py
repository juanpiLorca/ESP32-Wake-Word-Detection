import tensorflow as tf 

_SPEECH_DETECTION_CNN_MODEL = 'speech_commands_cnn_model'

def create_model(input_shape, output_dim, norm_layer=tf.keras.layers.Normalization()): 
    """
    Create a CNN model for speech command recognition.
    Args:
        input_shape (tuple): Shape of the input spectrogram (height, width, channels).
        --> input_shape is (49, 40, 1) for 1 second audio clips with 40 mel bins.
        output_dim (int): Number of output classes.
        norm_layer (tf.keras.layers.Layer): Normalization layer to apply to inputs.
    """
    return tf.keras.Sequential([
        tf.keras.layers.InputLayer(shape=input_shape),
        tf.keras.layers.Resizing(32, 32),
        norm_layer,
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(output_dim)
    ])

def create_dscnn_model(input_shape, output_dim):
    """
    Lightweight DS-CNN for embedded keyword spotting.
    Input shape should be (49, 10, 1) or (49, 40, 1).
    """
    return tf.keras.Sequential([
        tf.keras.layers.InputLayer(shape=input_shape),
        tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"),
        tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding="same", activation="relu"),
        tf.keras.layers.Conv2D(filters=64, kernel_size=1, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=2),
        tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding="same", activation="relu"),
        tf.keras.layers.Conv2D(filters=128, kernel_size=1, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=2),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(output_dim)
    ])


def make_esp32_keyword_model(input_shape, num_labels):
    """
    Attempt to replicate the DS-CNN model from the ESP32 keyword spotting example.
    - It must be time-invariant, so we use global average pooling.
    - Fewer parameters to fit on embedded device.
    """
    inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=4,
        strides=2,
        padding='same',
        use_bias=False)(inputs)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.DepthwiseConv2D(
        kernel_size=3,
        strides=1,
        padding='same',
        use_bias=False)(x)

    x = tf.keras.layers.Conv2D(
        filters=16,
        kernel_size=3,
        padding='same',
        use_bias=False)(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.DepthwiseConv2D(
        kernel_size=3,
        strides=1,
        padding='same',
        use_bias=False)(x)

    x = tf.keras.layers.Conv2D(
        filters=16,
        kernel_size=3,
        padding='same',
        use_bias=False)(x)
    x = tf.keras.layers.ReLU()(x)

    # GAP removes absolute time dimension
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)

    outputs = tf.keras.layers.Dense(num_labels)(x)

    return tf.keras.Model(inputs, outputs)

