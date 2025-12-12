import tensorflow as tf 

_SPEECH_DETECTION_CNN_MODEL = 'speech_commands_cnn_model'

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

