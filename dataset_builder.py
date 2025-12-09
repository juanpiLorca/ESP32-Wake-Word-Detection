"""SpeechCommands dataset builder."""

import pathlib
import tarfile

import pydub
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

_DATASET_NAME = "speech_commands"
_DOWNLOAD_PATH = (
    'http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz'
)
_TEST_DOWNLOAD_PATH = (
    'http://download.tensorflow.org/data/speech_commands_test_set_v0.02.tar.gz'
)

_DATASET_PATH = 'data/speech_commands'
_SPLITS = ['train', 'test']

WORDS = ['up', 'down']
SILENCE = '_silence_'
UNKNOWN = '_unknown_'
BACKGROUND_NOISE = '_background_noise_'
SAMPLE_RATE = 16000


class SpeechCommandsBuilder:
    """A simplified Speech Commands dataset builder."""

    VERSION = tfds.core.Version('0.0.3')
    RELEASE_NOTES = {
        '0.0.3': 'Fix audio data type with dtype=tf.int16.',
    }

    data_dir = pathlib.Path(_DATASET_PATH)

    def _info(self):
        """Dataset metadata."""
        return tfds.core.DatasetInfo(
            builder=self,
            features=tfds.features.FeaturesDict({
                'audio': tfds.features.Audio(
                    file_format='wav',
                    sample_rate=SAMPLE_RATE,
                    dtype=np.int16
                ),
                'label': tfds.features.ClassLabel(
                    names=WORDS + [SILENCE, UNKNOWN]
                ),
            }),
            supervised_keys=('audio', 'label'),
            homepage="https://arxiv.org/abs/1804.03209",
        )

    def _download(self):
        """Download tar.gz archives."""

        print("Downloading dataset...")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        train_tar_path = tf.keras.utils.get_file(
            fname=_DATASET_NAME + "_train.tar.gz",
            origin=_DOWNLOAD_PATH,
            extract=False,
            cache_dir='.',
            cache_subdir="speech_commands"
        )

        test_tar_path = tf.keras.utils.get_file(
            fname=_DATASET_NAME + "_test.tar.gz",
            origin=_TEST_DOWNLOAD_PATH,
            extract=False,
            cache_dir='.',
            cache_subdir="speech_commands"
        )

        return train_tar_path, test_tar_path

    def _extract_tar(self, tar_path, output_path):
        """Extract tar.gz to a directory."""
        print(f"Extracting {tar_path}...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=output_path)

    def _create_structure(self):
        """Create clean folder tree: data/speech_commands/{train,test}/{classes}"""
        for split in _SPLITS:
            for cls in WORDS + [SILENCE, UNKNOWN]:
                dir_path = self.data_dir / split / cls
                dir_path.mkdir(parents=True, exist_ok=True)

    def _filter_classes(self, extracted_root, split):
        """Process wav files into output directory."""

        split_root = self.data_dir / split

        for folder in extracted_root.iterdir():
            if not folder.is_dir():
                continue

            class_name = folder.name

            if class_name == BACKGROUND_NOISE:
                for file in folder.glob('*.wav'):

                    audio_samples = np.array(
                        pydub.AudioSegment.from_file(
                            file, format='wav'
                        ).get_array_of_samples()
                    )

                    step = SAMPLE_RATE // 2  # 0.5 second stride
                    for idx, start in enumerate(
                        range(0, len(audio_samples) - SAMPLE_RATE, step)
                    ):
                        audio_segment = audio_samples[start:start + SAMPLE_RATE]

                        target_file = split_root / SILENCE / f"{class_name}_{idx}.wav"

                        pydub.AudioSegment(
                            audio_segment.tobytes(),
                            frame_rate=SAMPLE_RATE,
                            sample_width=audio_segment.dtype.itemsize,
                            channels=1,
                        ).export(target_file, format='wav')

                continue  # Done with _background_noise_

            if class_name == SILENCE: # only for test set
                target_dir = split_root / class_name
            else:
                if class_name in WORDS:
                    target_dir = split_root / class_name
                else:
                    target_dir = split_root / UNKNOWN

            target_dir.mkdir(parents=True, exist_ok=True)

            # Copy wav files
            for wav_file in folder.glob("*.wav"):
                out_path = target_dir / wav_file.name
                pydub.AudioSegment.from_file(wav_file).export(out_path, format="wav")

    def _build(self):
        """Main entry point: download, extract, structure, filter."""

        print("=== Building Speech Commands Subset Dataset ===")
        self._create_structure()

        train_tar, test_tar = self._download()

        extract_root = pathlib.Path("speech_commands/raw")
        extract_root.mkdir(parents=True, exist_ok=True)

        train_dir = extract_root / "train"
        test_dir = extract_root / "test"
        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)

        self._extract_tar(train_tar, train_dir)
        self._extract_tar(test_tar, test_dir)
        print("Processing TRAIN split...")
        self._filter_classes(train_dir, "train")

        print("Processing TEST split...")
        self._filter_classes(test_dir, "test")

        print("Dataset ready at:", self.data_dir)


if __name__ == "__main__":
    builder = SpeechCommandsBuilder()
    builder._build()

    

        
    