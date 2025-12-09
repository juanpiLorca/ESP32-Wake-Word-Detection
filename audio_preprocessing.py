"""Audio preprocessing utilities for adding noise to audio data."""

import os
import random
import pydub
import numpy as np 
import matplotlib.pyplot as plt

_MAIN_DIR = 'data/speech_commands/'
_SILENCE_BACKGROUND = _MAIN_DIR + 'train/_silence_/'
_SILENCE_BACKGROUND_TEST = _MAIN_DIR + 'test/_silence_/'


def add_noise_to_audio(real_path, noise_files, noise_level=(0.0, 0.1)):
    """Add random background noise to an audio file and return noisy PCM16 array."""

    # ----- Load real audio -----
    audio = pydub.AudioSegment.from_file(real_path, format='wav')
    audio_np = np.array(audio.get_array_of_samples(), dtype=np.float32)
    audio_np = audio_np / 32768.0  # PCM16 → float [-1,1]

    # ----- Pick a random noise file -----
    noise_path = random.choice(noise_files)
    noise = pydub.AudioSegment.from_file(noise_path, format='wav')
    noise_np = np.array(noise.get_array_of_samples(), dtype=np.float32)
    noise_np = noise_np / 32768.0

    # Match lengths (loop or crop noise)
    if len(noise_np) < len(audio_np):
        repeats = (len(audio_np) // len(noise_np)) + 1
        noise_np = np.tile(noise_np, repeats)
    noise_np = noise_np[:len(audio_np)]

    # ----- Random noise gain -----
    gain = random.uniform(noise_level[0], noise_level[1])

    # ----- Mix -----
    noisy = audio_np + gain * noise_np

    # ----- Clip and convert back to PCM16 -----
    noisy_pcm16 = np.clip(noisy * 32768.0, -32768, 32767).astype(np.int16)

    return noisy_pcm16, audio.frame_rate


def load_noise_files(noise_dir):
    return [
        os.path.join(noise_dir, f)
        for f in os.listdir(noise_dir)
        if f.endswith('.wav')
    ]

def generate_noisy_dataset(src_root, dst_root, noise_dir):
    os.makedirs(dst_root, exist_ok=True)

    noise_files = load_noise_files(noise_dir)

    classes = ["on", "off", "_unknown_"]

    for cls in classes:
        src_dir = os.path.join(src_root, cls)
        dst_dir = os.path.join(dst_root, cls)
        os.makedirs(dst_dir, exist_ok=True)

        print(f"Processing class: {cls}")

        for fname in os.listdir(src_dir):
            if not fname.endswith(".wav"):
                continue

            real_path = os.path.join(src_dir, fname)

            noisy_pcm16, sr = add_noise_to_audio(real_path, noise_files)

            # Save WAV via pydub
            noisy_audio = pydub.AudioSegment(
                noisy_pcm16.tobytes(),
                frame_rate=sr,
                sample_width=2,    # PCM16
                channels=1
            )

            noisy_audio.export(os.path.join(dst_dir, fname), format="wav")


if __name__ == "__main__":

    _SRC_ROOT = _MAIN_DIR + 'train/'
    _DST_ROOT = _SRC_ROOT

    generate_noisy_dataset(
        src_root=_SRC_ROOT,
        dst_root=_DST_ROOT,
        noise_dir=_SILENCE_BACKGROUND
    )