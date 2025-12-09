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
    return tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),  # e.g., 49 × 10 × 1
        tf.keras.layers.Conv2D(8, 3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),   # Output: 24 × 20 × 8
        tf.keras.layers.DepthwiseConv2D(3, activation='relu', padding='same'),
        tf.keras.layers.Conv2D(16, 1, activation='relu'),   # Pointwise conv
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),     # Output: 12 × 10 × 16
        tf.keras.layers.DepthwiseConv2D(3, activation='relu', padding='same'),
        tf.keras.layers.Conv2D(32, 1, activation='relu'),   # Output: 12 × 10 × 32
        tf.keras.layers.GlobalAveragePooling2D(),           # Output: 32
        tf.keras.layers.Dense(num_labels)
    ])
