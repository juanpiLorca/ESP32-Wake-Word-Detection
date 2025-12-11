import os
import math
import pydub

import numpy as np 
import tensorflow as tf
from scipy.fftpack import dct

from tensorflow.python.ops import gen_audio_ops as audio_ops
from tensorflow.python.platform import gfile

def pre_emphasis_filter(signal, pre_emphasis_coeff=0.97):
  """Applies a pre-emphasis filter on the input signal.

  Args:
    signal: 1D numpy array of the audio signal.
    pre_emphasis_coeff: Pre-emphasis coefficient.
  """
  emp_signal = np.append(signal[0], signal[1:] - pre_emphasis_coeff * signal[:-1])
  return emp_signal

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

def compute_audio_mfcc_features(audio_data, model_settings):
  # Params definition 
  frame_size_length = model_settings['window_size_samples']
  frame_stride_length = model_settings['window_stride_samples']
  
  nfft = 512
  nfilt = 40
  num_ceps = 12
  low_freq_mel = 0
  high_freq_mel = (2595 * np.log10(1 + (model_settings['sample_rate'] / 2) / 700))  # Convert Hz to Mel
  mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
  hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
  bin = np.floor((nfft + 1) * hz_points / model_settings['sample_rate'])

  x_emp = pre_emphasis_filter(audio_data)
  signal_length = len(x_emp)

  num_frames = int(np.ceil(float(np.abs(signal_length - frame_size_length)) / frame_stride_length))  
  pad_signal_length = num_frames * frame_stride_length + frame_size_length
  z = np.zeros(shape=((pad_signal_length - signal_length),))
  pad_signal = np.append(x_emp, z)
  indices = np.tile(np.arange(0, frame_size_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_stride_length, frame_stride_length), (frame_size_length, 1)).T
  frames = pad_signal[indices.astype(np.int32, copy=False)]

  hamming_window = np.hamming(frame_size_length)
  frames_windowed = frames * hamming_window

  mag_frames = np.absolute(np.fft.rfft(frames_windowed, n=nfft))
  pow_frames = ((1.0 / nfft) * (mag_frames ** 2))
  fbank = gen_filter_banks(nfilt, nfft, bin)
  filter_banks = np.dot(pow_frames, fbank.T)
  filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
  filter_banks = 20 * np.log10(filter_banks)  # dB
  mfcc_norm = (filter_banks - (np.mean(filter_banks, axis=0) + 1e-8))# / (np.std(filter_banks, axis=0) + 1e-8)
  # mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1:(num_ceps+1)]
  # mfcc_norm = (mfcc - mfcc.mean(axis=0) + 1e-8) 
  return mfcc_norm, pow_frames


class AudioProcessor: 

  output_ = None
  background_data_ = []

  def prepare_background_data(self, background_dir):
    """Searches a folder for background noise audio, and loads it into memory.

    It's expected that the background audio samples will be in a subdirectory
    named '_background_noise_' inside the 'data_dir' folder, as .wavs that match
    the sample rate of the training data, but can be much longer in duration.

    If the '_background_noise_' folder doesn't exist at all, this isn't an
    error, it's just taken to mean that no background noise augmentation should
    be used. If the folder does exist, but it's empty, that's treated as an
    error.

    Returns:
      List of raw PCM-encoded audio samples of background noise.

    Raises:
      Exception: If files aren't found in the folder.
    """
    if not os.path.exists(background_dir):
      print("No background noise directory found. Skipping noise augmentation.")
      return []

    noise_files = [
        os.path.join(background_dir, f)
        for f in os.listdir(background_dir)
        if f.endswith('.wav')
    ]

    if len(noise_files) == 0:
        raise Exception("Background noise directory exists but contains no .wav files.")

    print(f"Loading {len(noise_files)} background noise files...")

    for nf in noise_files:
        audio_seg = pydub.AudioSegment.from_file(nf, format='wav')
        samples = np.array(audio_seg.get_array_of_samples(), dtype=np.float32)
        samples /= 32768.0  # Convert to [-1.0, 1.0]
        self.background_data_.append(samples)
  
    return self.background_data_

  def get_features_from_audio(self, audio_data, model_settings): 

    if model_settings['preprocessing_imp'] == 'tensorflow': 
      spectrogram = audio_ops.audio_spectrogram(
          audio_data,
          window_size=model_settings['window_size_samples'],
          stride=model_settings['window_stride_samples'],
          magnitude_squared=True
      )
      # The number of buckets in each FFT row in the spectrogram will depend on
      # how many input samples there are in each window. This can be quite
      # large, with a 160 sample window producing 127 buckets for example. We
      # don't need this level of detail for classification, so we often want to
      # shrink them down to produce a smaller result. That's what this section
      # implements. One method is to use average pooling to merge adjacent
      # buckets, but a more sophisticated approach is to apply the MFCC
      # algorithm to shrink the representation.
      if model_settings['preprocess'] == 'average':
        self.output_ = tf.nn.pool(
              input=tf.expand_dims(spectrogram, -1),
              window_shape=[1, model_settings['average_window_width']],
              strides=[1, model_settings['average_window_width']],
              pooling_type='AVG',
              padding='SAME'
            )
      
      elif model_settings['preprocess'] == 'mfcc':
        self.output_ = audio_ops.mfcc(
              spectrogram=spectrogram,
              sample_rate=model_settings['sample_rate'],
              dct_coefficient_count=model_settings['fingerprint_width']
            )
        # MFCC doesn't accept a 3D tensor, but a 2D one, so we need to add an extra axis
        self.output_ = tf.expand_dims(self.output_, -1)
      
      else: 
        raise ValueError('Unknown preprocess mode "%s" (should be "mfcc" or'
                        ' "average")' % (model_settings['preprocess']))
    else:
      self.output_, spectrogram = compute_audio_mfcc_features(
          audio_data.numpy(), model_settings)
      self.output_ = tf.convert_to_tensor(self.output_, dtype=tf.float32)
      self.output_ = tf.expand_dims(self.output_, -1)  # Add channel dimension
    return self.output_, spectrogram


def _next_power_of_two(x):
  """Calculates the smallest enclosing power of two for an input.

  Args:
    x: Positive float or integer number.

  Returns:
    Next largest power of two integer.
  """
  return 1 if x == 0 else 2**(int(x) - 1).bit_length()


def prepare_model_settings(label_count, preprocess_imp, sample_rate, clip_duration_ms,
                           window_size_ms, window_stride_ms, feature_bin_count,
                           preprocess):
  """Calculates common settings needed for all models.

  Args:
    label_count: How many classes are to be recognized.
    sample_rate: Number of audio samples per second.
    clip_duration_ms: Length of each audio clip to be analyzed.
    window_size_ms: Duration of frequency analysis window.
    window_stride_ms: How far to move in time between frequency windows.
    feature_bin_count: Number of frequency bins to use for analysis.
    preprocess: How the spectrogram is processed to produce features.

  Returns:
    Dictionary containing common settings.

  Raises:
    ValueError: If the preprocessing mode isn't recognized.
  """
  desired_samples = int(sample_rate * clip_duration_ms / 1000)
  window_size_samples = int(sample_rate * window_size_ms / 1000)
  window_stride_samples = int(sample_rate * window_stride_ms / 1000)
  length_minus_window = (desired_samples - window_size_samples)
  if length_minus_window < 0:
    spectrogram_length = 0
  else:
    spectrogram_length = 1 + int(length_minus_window / window_stride_samples)
  if preprocess == 'average':
    fft_bin_count = 1 + (_next_power_of_two(window_size_samples) / 2)
    average_window_width = int(math.floor(fft_bin_count / feature_bin_count))
    fingerprint_width = int(math.ceil(fft_bin_count / average_window_width))
  elif preprocess == 'mfcc':
    average_window_width = -1
    fingerprint_width = feature_bin_count
  elif preprocess == 'micro':
    average_window_width = -1
    fingerprint_width = feature_bin_count
  else:
    raise ValueError('Unknown preprocess mode "%s" (should be "mfcc",'
                     ' "average", or "micro")' % (preprocess))
  fingerprint_size = fingerprint_width * spectrogram_length
  return {
      'preprocessing_imp': preprocess_imp,
      'sample_rate': sample_rate,
      'desired_samples': desired_samples,
      'window_size_samples': window_size_samples,
      'window_stride_samples': window_stride_samples,
      'spectrogram_length': spectrogram_length,
      'fingerprint_width': fingerprint_width,
      'fingerprint_size': fingerprint_size,
      'label_count': label_count,
      'sample_rate': sample_rate,
      'preprocess': preprocess,
      'average_window_width': average_window_width,
  }

def get_features_range(model_settings):
  """Returns the expected min/max for generated features.

  Args:
    model_settings: Information about the current model being trained.

  Returns:
    Min/max float pair holding the range of features.

  Raises:
    Exception: If preprocessing mode isn't recognized.
  """
  if model_settings['preprocess'] == 'average':
    features_min = 0.0
    features_max = 127.5
  elif model_settings['preprocess'] == 'mfcc':
    features_min = -247.0
    features_max = 30.0
  else:
    raise Exception('Unknown preprocess mode "%s" (should be "mfcc",'
                    ' " or average")' % (model_settings['preprocess']))
  return features_min, features_max

def wav_to_features(sample_rate, preprocess_imp, clip_duration_ms, window_size_ms,
                    window_stride_ms, feature_bin_count, preprocess,
                    input_audio, spectrogram_output=False):
  """Converts an audio file into its corresponding feature map.

  Args:
    sample_rate: Expected sample rate of the wavs.
    clip_duration_ms: Expected duration in milliseconds of the wavs.
    window_size_ms: How long each spectrogram timeslice is.
    window_stride_ms: How far to move in time between spectrogram timeslices.
    feature_bin_count: How many bins to use for the feature fingerprint.
    quantize: Whether to train the model for eight-bit deployment.
    preprocess: Spectrogram processing mode; "mfcc", "average" or "micro".
    input_wav: Path to the audio WAV file to read.
    output_c_file: Where to save the generated C source file.
  """
  model_settings = prepare_model_settings(
      0, preprocess_imp, sample_rate, clip_duration_ms, window_size_ms, window_stride_ms,
      feature_bin_count, preprocess)
  audio_processor = AudioProcessor()
  
  features, spectrogram = audio_processor.get_features_from_audio(input_audio, model_settings)
  if model_settings['preprocessing_imp'] == 'tensorflow':
    features = features[0]
    spectrogram = spectrogram[0]

  if spectrogram_output:
    return features, spectrogram
  else:
    return features
  
def wav_to_features_c_file(sample_rate, clip_duration_ms, window_size_ms,
                           window_stride_ms, feature_bin_count, preprocess,
                           input_wav, output_c_file, quantize=False):
  """Converts an audio file into its corresponding feature map and saves it as a C source file.

  Args:
    sample_rate: Expected sample rate of the wavs.
    clip_duration_ms: Expected duration in milliseconds of the wavs.
    window_size_ms: How long each spectrogram timeslice is.
    window_stride_ms: How far to move in time between spectrogram timeslices.
    feature_bin_count: How many bins to use for the feature fingerprint.
    quantize: Whether to train the model for eight-bit deployment.
    preprocess: Spectrogram processing mode; "mfcc", "average" or "micro".
    input_wav: Path to the audio WAV file to read.
    output_c_file: Where to save the generated C source file.
  """
  input_audio =pydub.AudioSegment.from_file(input_wav, format='wav')
  input_audio = np.array(input_audio.get_array_of_samples(), dtype=np.float32)
  input_audio /= 32768.0  # Convert to [-1.0, 1.0]
  input_audio = tf.convert_to_tensor(input_audio, dtype=tf.float32)
  input_audio = tf.expand_dims(input_audio, -1)  # Add batch dimension

  model_settings = prepare_model_settings(
      0, sample_rate, clip_duration_ms, window_size_ms, window_stride_ms,
      feature_bin_count, preprocess)
  audio_processor = AudioProcessor()
  
  results, _ = audio_processor.get_features_from_audio(input_audio, model_settings)
  features = results[0]
  features = results[..., 0].numpy()  # Remove channel dimension

  variable_base = os.path.splitext(os.path.basename(output_c_file))[0]

  # Save a C source file containing the feature data as an array.
  with gfile.GFile(output_c_file, 'w') as f:
    f.write('/* File automatically created by\n')
    f.write(' * project_speech_commands/core.py \\\n')
    f.write(' * --sample_rate=%d \\\n' % sample_rate)
    f.write(' * --clip_duration_ms=%d \\\n' % clip_duration_ms)
    f.write(' * --window_size_ms=%d \\\n' % window_size_ms)
    f.write(' * --window_stride_ms=%d \\\n' % window_stride_ms)
    f.write(' * --feature_bin_count=%d \\\n' % feature_bin_count)
    if quantize:
      f.write(' * --quantize=1 \\\n')
    f.write(' * --preprocess="%s" \\\n' % preprocess)
    f.write(' * --input_wav="%s" \\\n' % input_wav)
    f.write(' * --output_c_file="%s" \\\n' % output_c_file)
    f.write(' */\n\n')
    f.write('const int g_%s_width = %d;\n' %
            (variable_base, model_settings['fingerprint_width']))
    f.write('const int g_%s_height = %d;\n' %
            (variable_base, model_settings['spectrogram_length']))
    if quantize:
      features_min, features_max = get_features_range(model_settings)
      f.write('const unsigned char g_%s_data[] = {' % variable_base)
      i = 0
      for value in features.flatten():
        quantized_value = int(
            round(
                (255 * (value - features_min)) / (features_max - features_min)))
        if quantized_value < 0:
          quantized_value = 0
        if quantized_value > 255:
          quantized_value = 255
        if i == 0:
          f.write('\n  ')
        f.write('%d, ' % (quantized_value))
        i = (i + 1) % 10
    else:
      f.write('const float g_%s_data[] = {\n' % variable_base)
      i = 0
      for value in features.flatten():
        if i == 0:
          f.write('\n  ')
        f.write('%f, ' % value)
        i = (i + 1) % 10
    f.write('\n};\n')


if __name__ == "__main__":
  sample_rate = 16000
  clip_duration_ms = 1000
  window_size_ms = 30.0
  window_stride_ms = 20.0
  feature_bin_count = 40
  preprocess = 'mfcc'

  input_wav = 'data/speech_commands/test/up/0c40e715_nohash_0.wav'
  output_c_file = 'up_features_data_quant.cc'
  wav_to_features_c_file(
      sample_rate, clip_duration_ms, window_size_ms, window_stride_ms,
      feature_bin_count, preprocess, input_wav, output_c_file, quantize=True)
  
  print(f'Wrote feature data to {output_c_file}')

  input_wav = 'data/speech_commands/test/down/0c40e715_nohash_0.wav'
  output_c_file = 'down_features_data_quant.cc'
  wav_to_features_c_file(
      sample_rate, clip_duration_ms, window_size_ms, window_stride_ms,
      feature_bin_count, preprocess, input_wav, output_c_file, quantize=True)
  
  print(f'Wrote feature data to {output_c_file}')