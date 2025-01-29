import os
import random
from collections import defaultdict
from glob import glob

import librosa
import numpy as np
import tensorflow as tf
import torchaudio
from pedalboard import (
    Gain,
    HighpassFilter,
    LowpassFilter,
    Pedalboard,
    PitchShift,
    Reverb,
)

ALL_EFFECTS = ["pitch", "reverb", "eq", "noise", "stretch"]

default_adversarial_params = {
    "adversarial_pitch": 2,
    "adversarial_stretch": 0.2,  # -> [80%, 120%]
    "adversarial_noise_std": 0.01,
    "sr": 44100,
    "target_audio_slice": 0.78,
}

default_loader_params = {
    "sr": 44100,  # sampling rate
    "batch_size": 32,
    "repeat": 5,  # repeat opened file in next batches for efficiency
    "shuffle": 2,  # factor of batch size in buffer
    "seed": 123,
    "audio_slice": 3.0,  # in seconds
}

default_augmenter_params = {
    "sr": 44100,
    "fft": {
        "win": 512,
        "hop": 256,
        "n_fft": 1024,
    },
    "normalise_mean": 0.0,
    "normalise_std": 1.0,
    "hf_cut": 16000,
    "lf_cut": 4000,
    "effects": [],
}


class AudioLoader:
    def __init__(
        self, real_db_path, fake_db_path, params={}, split_path=None, codec=""
    ):
        """
        Assumptions:
        - real_db is structured as folder/songs.mp3
        - fake_db is structured as encoder/folder/songs.mp3
        """

        self.params = default_loader_params
        self.params.update(params)

        self.pos_list = glob(os.path.join(real_db_path, "**/*.mp3"))
        self.neg_list = glob(os.path.join(fake_db_path, "**/*.mp3"))
        # if codec != "":
        #     self.pos_list = glob(os.path.join(real_db_path, "*.{}".format(codec)))
        #     self.neg_list = glob(os.path.join(real_db_path, "*.{}".format(codec)))

        if len(self.pos_list) * len(self.neg_list) == 0:
            raise ValueError(
                "DB path incorrect, no file found: {}, {}".format(
                    real_db_path, fake_db_path
                )
            )

        self.split_mp3 = {
            "train": [],
            "validation": [],
            "test": [],
        }
        self.dict_pos = {}
        set_pos = set()
        for s_path in self.pos_list:
            s_mp3 = s_path.split("/")[-1]
            s_mp3 = s_mp3.split(".")[0] + ".mp3"  # fix for other codec...
            self.dict_pos[s_mp3] = s_path
            set_pos.add(s_mp3)
            if "train" in s_path:
                self.split_mp3["train"].append(s_mp3)
            elif "test" in s_path:
                self.split_mp3["test"].append(s_mp3)

        self.dict_pos = tf.lookup.StaticHashTable(
            initializer=tf.lookup.KeyValueTensorInitializer(
                keys=tf.constant(list(self.dict_pos.keys())),
                values=tf.constant(list(self.dict_pos.values())),
            ),
            default_value=tf.constant("NO-FILE"),
        )

        self.dict_neg = defaultdict(dict)
        set_neg = defaultdict(set)
        for s_path in self.neg_list:
            s_path_split = s_path.split("/")
            s_mp3 = s_path_split[-1]
            s_mp3 = s_mp3.split(".")[0] + ".mp3"  # fix for other codec...
            encoder = s_path_split[-3]
            if codec != "":
                encoder = s_path_split[-2]
            if encoder == "real":
                continue
            self.dict_neg[encoder + "." + s_mp3] = s_path
            set_neg[encoder].add(s_mp3)
            if "test" in s_path:
                self.split_mp3["test"].append(encoder + "." + s_mp3)
            if "train" in s_path:
                self.split_mp3["train"].append(encoder + "." + s_mp3)

        # shuffle train and move 20% to validation
        np.random.seed(self.params["seed"])
        tf.random.set_seed(self.params["seed"])

        np.random.shuffle(self.split_mp3["train"])
        n_val = int(len(self.split_mp3["train"]) * 0.2)
        self.split_mp3["validation"] = self.split_mp3["train"][:n_val]
        self.split_mp3["train"] = self.split_mp3["train"][n_val:]

        self.encoders = sorted(list(set_neg.keys()))
        self.n_encoders = len(self.encoders)
        self.dict_encoder = tf.lookup.StaticHashTable(
            initializer=tf.lookup.KeyValueTensorInitializer(
                keys=tf.constant(list(range(self.n_encoders))),
                values=tf.constant(self.encoders),
            ),
            default_value=tf.constant("NO-ENCODER"),
        )

        self.dict_neg = tf.lookup.StaticHashTable(
            initializer=tf.lookup.KeyValueTensorInitializer(
                keys=tf.constant(list(self.dict_neg.keys())),
                values=tf.constant(list(self.dict_neg.values())),
            ),
            default_value=tf.constant("NO-FILE"),
        )

        intersection_mp3 = set_pos
        for encoder in set_neg:
            for sample in set_neg[encoder]:
                intersection_mp3.add(encoder + "." + sample)
        self.all_mp3 = list(intersection_mp3)

    def gen_labels(self, buffer_label=10000):
        """generate fake/true labels and encoder labels"""
        y_iterator_1 = tf.data.Dataset.from_tensor_slices(
            tf.convert_to_tensor([0, 1], tf.float32)  # alternate
        ).repeat()
        y_iterator_2 = tf.data.Dataset.from_tensor_slices(
            tf.random.uniform((buffer_label,), 0, self.n_encoders, tf.int32)
        ).repeat()
        y_iterator = tf.data.Dataset.zip((y_iterator_1, y_iterator_2))
        return y_iterator

    def gen_encoder_labels(self, encoder):
        """generate fake/true labels and encoder labels"""
        if encoder not in self.encoders:
            raise ValueError("Encoder unavailable:", encoder)

        y_iterator_1 = tf.data.Dataset.from_tensor_slices(
            tf.convert_to_tensor([0, 1], tf.float32)  # alternate
        ).repeat()
        idx_encoder = self.encoders.index(encoder)
        y_iterator_2 = tf.data.Dataset.from_tensor_slices(
            tf.constant([idx_encoder], tf.int32)
        ).repeat()
        y_iterator = tf.data.Dataset.zip((y_iterator_1, y_iterator_2))
        return y_iterator

    def get_file_path(self, mp3, labels):
        y, idx_encoder = labels
        encoder = self.dict_encoder.lookup(idx_encoder)

        if y > 0:
            return self.dict_pos.lookup(mp3), labels
        return self.dict_neg.lookup(mp3), labels

    @tf.function(reduce_retracing=True)
    def torch_open_audio(self, f_name, labels):
        def py_open_audio(fpath):
            audio_raw, sr = torchaudio.load(
                fpath.numpy().decode("utf-8"),
                channels_first=False,
            )
            audio_raw = audio_raw.numpy()
            return audio_raw

        audio = tf.py_function(
            py_open_audio,
            [f_name],
            tf.float32,
        )

        return audio, labels

    @tf.function
    def get_pair_audio(self, mp3, labels):
        y, idx_encoder = labels
        real_path, _ = self.get_file_path(mp3, (tf.ones(1), idx_encoder))
        fake_path, _ = self.get_file_path(mp3, (tf.zeros(1), idx_encoder))

        real_audio, _ = self.torch_open_audio(real_path, labels)
        fake_audio, _ = self.torch_open_audio(fake_path, labels)
        min_shape = tf.minimum(tf.shape(real_audio)[0], tf.shape(fake_audio)[0])
        fake_audio = tf.slice(fake_audio, (0, 0), (min_shape, 2))
        real_audio = tf.slice(real_audio, (0, 0), (min_shape, 2))
        return tf.stack((real_audio, fake_audio), 0)

    @tf.function
    def pair_mixing(self, n_bins, audios):
        labels = tf.cast(tf.linspace(0, 1, n_bins), tf.float32)
        mixing = tf.stack((labels, 1 - labels), 1)

        return tf.tensordot(mixing, audios, 1), labels

    @tf.function
    def slice_audio(self, x, labels):
        """assumes channels last"""
        target_len = int(self.params["audio_slice"] * self.params["sr"])
        max_offset = tf.shape(x)[-2] - target_len
        offset = tf.random.uniform((), maxval=max_offset + 1, dtype=tf.int32)
        return tf.slice(x, (offset, 0), (target_len, 2)), labels

    @tf.function
    def patch_spec(self, x, labels):
        """cut into a spectrogram"""
        x_shape = tf.shape(x)
        max_offset_t = x_shape[0] - self.params["patch_size_t"]
        max_offset_f = x_shape[1] - self.params["patch_size_f"]
        min_offset_f = 0
        if "patch_f_min_threshold" in self.params:
            min_offset_f = self.params["patch_f_min_threshold"]
        if "patch_f_max_threshold" in self.params:
            max_offset_f = (
                self.params["patch_f_max_threshold"] - self.params["patch_size_f"]
            )
        offset_t = tf.random.uniform((), maxval=max_offset_t + 1, dtype=tf.int32)
        offset_f = tf.random.uniform(
            (), minval=min_offset_f, maxval=max_offset_f + 1, dtype=tf.int32
        )
        return tf.slice(
            x,
            (offset_t, offset_f, 0),
            (self.params["patch_size_t"], self.params["patch_size_f"], -1),
        ), labels

    @tf.function
    def patch_batch_spec(self, x, labels):
        """same but random batch"""
        x_shape = tf.shape(x)
        max_offset_t = x_shape[1] - self.params["patch_size_t"]
        patch_size_f = tf.random.uniform(
            (),
            minval=self.params["patch_size_f_min"],
            maxval=self.params["patch_size_f"] + 1,
            dtype=tf.int32,
        )

        max_offset_f = x_shape[2] - patch_size_f
        offset_t = tf.random.uniform((), maxval=max_offset_t + 1, dtype=tf.int32)
        offset_f = tf.random.uniform((), maxval=max_offset_f + 1, dtype=tf.int32)
        return (
            tf.slice(
                x,
                (0, offset_t, offset_f, 0),
                (-1, self.params["patch_size_t"], patch_size_f, -1),
            ),
            labels,
        )

    def create_tf_iterator(
        self, mode="train", augmenter=None, encoder=None, adversarial=None
    ):
        # self.params["audio_slice"] = 0.1
        iterator = tf.data.Dataset.from_tensor_slices(self.split_mp3[mode])
        # list(iterator.as_numpy_iterator())
        if mode != "test":
            iterator = iterator.repeat()
        if not encoder:
            # y_iterator = self.gen_labels()
            # create new function that outputs 0 if sample starts with 0., 1 otherwise
            labels = []
            encoders = []
            for sample in self.split_mp3[mode]:
                if sample.startswith("0."):
                    labels.append(0)
                else:
                    labels.append(1)
                encoders.append(0)
            y_iterator_1 = tf.data.Dataset.from_tensor_slices(labels)
            y_iterator_2 = tf.data.Dataset.from_tensor_slices(
                tf.constant(encoders, tf.int32)
            )
            if mode != "test":
                y_iterator_1 = y_iterator_1.repeat()
                y_iterator_2 = y_iterator_2.repeat()
            y_iterator = tf.data.Dataset.zip((y_iterator_1, y_iterator_2))
        else:
            y_iterator = self.gen_encoder_labels(encoder)

        iterator = tf.data.Dataset.zip((iterator, y_iterator))
        iterator = iterator.map(self.get_file_path)

        iterator = iterator.map(
            self.torch_open_audio, num_parallel_calls=tf.data.AUTOTUNE
        )
        if mode != "test":
            iterator = iterator.flat_map(
                lambda *x: tf.data.Dataset.from_tensors(x).repeat(self.params["repeat"])
            )

        if adversarial:
            if adversarial.effects != ["stretch"]:
                iterator = iterator.map(self.slice_audio)
            iterator = iterator.batch(
                self.params["batch_size"],
                num_parallel_calls=tf.data.AUTOTUNE,
            )
            iterator = iterator.map(
                lambda x, y: (adversarial.batch_transform(x), y),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
            iterator = iterator.unbatch()
            if adversarial.effects == ["stretch"]:
                iterator = iterator.map(self.slice_audio)
        else:
            iterator = iterator.map(self.slice_audio)

        if augmenter:
            iterator = iterator.map(lambda x, y: (augmenter.transform(x), y))

        if mode != "test":
            iterator = iterator.shuffle(
                self.params["batch_size"]
                * self.params["repeat"]
                * self.params["shuffle"]
            )

        iterator = iterator.batch(
            self.params["batch_size"],
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        iterator = iterator.prefetch(tf.data.AUTOTUNE)

        return iterator

    def create_fast_eval_iterator(
        self, encoder="real", mode="test", augmenter=None, adversarial=None
    ):
        """batch adversarial"""

        iterator = tf.data.Dataset.from_tensor_slices(self.split_mp3[mode])
        if encoder == "real":
            y_iterator_1 = tf.data.Dataset.from_tensor_slices(
                tf.ones((1,), tf.float32)
            ).repeat()
            y_iterator_2 = tf.data.Dataset.from_tensor_slices(
                tf.constant([-1], tf.int32)
            ).repeat()
        else:
            if encoder not in self.encoders:
                raise ValueError("Encoder unavailable:", encoder)
            idx_encoder = self.encoders.index(encoder)
            y_iterator_1 = tf.data.Dataset.from_tensor_slices(
                tf.zeros((1,), tf.float32)
            ).repeat()
            y_iterator_2 = tf.data.Dataset.from_tensor_slices(
                tf.constant([idx_encoder], tf.int32)
            ).repeat()
        y_iterator = tf.data.Dataset.zip((y_iterator_1, y_iterator_2))

        iterator = tf.data.Dataset.zip((iterator, y_iterator))
        iterator = iterator.map(self.get_file_path)
        iterator = iterator.map(
            self.torch_open_audio,  # self.open_audio
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        iterator = iterator.flat_map(
            lambda *x: tf.data.Dataset.from_tensors(x).repeat(self.params["repeat"])
        )
        iterator = iterator.map(self.slice_audio)
        iterator = iterator.batch(
            self.params["batch_size"],
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        if adversarial:
            iterator = iterator.map(
                lambda x, y: (adversarial.batch_transform(x), y),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
        iterator = iterator.unbatch()
        if augmenter:
            iterator = iterator.map(lambda x, y: (augmenter.transform(x), y))
        iterator = iterator.batch(
            self.params["batch_size"],
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        iterator = iterator.prefetch(tf.data.AUTOTUNE)

        return iterator

    def create_calibration_iterator(self, mode="test", n_bins=10, augmenter=None):
        """only used for computing the model calibration, tests different audio mix"""
        iterator = tf.data.Dataset.from_tensor_slices(self.split_mp3[mode])
        y_iterator = self.gen_labels()  # but ignore y[0]

        iterator = tf.data.Dataset.zip((iterator, y_iterator))
        iterator = iterator.map(
            self.get_pair_audio,
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        iterator = iterator.map(lambda x: self.pair_mixing(n_bins, x))
        iterator = iterator.unbatch()
        iterator = iterator.map(self.slice_audio)
        if augmenter:
            iterator = iterator.map(lambda x, y: (augmenter.transform(x), y))
        iterator = iterator.batch(
            self.params["batch_size"],
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        iterator = iterator.prefetch(tf.data.AUTOTUNE)

        return iterator

    def create_patch_iterator(self, augmenter, mode="train"):
        """Train the patch model for spectrogram interpretability"""
        iterator = tf.data.Dataset.from_tensor_slices(self.split_mp3[mode])
        iterator = iterator.repeat()
        y_iterator = self.gen_labels()

        iterator = tf.data.Dataset.zip((iterator, y_iterator))
        iterator = iterator.map(self.get_file_path)

        iterator = iterator.map(
            self.torch_open_audio,  # self.open_audio
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        iterator = iterator.map(self.slice_audio)  # accelerate before stft
        iterator = iterator.map(lambda x, y: (augmenter.transform(x), y))
        iterator = iterator.flat_map(
            lambda *x: tf.data.Dataset.from_tensors(x).repeat(self.params["repeat"])
        )
        iterator = iterator.map(self.patch_spec)
        iterator = iterator.shuffle(
            self.params["batch_size"] * self.params["repeat"] * self.params["shuffle"]
        )
        iterator = iterator.batch(
            self.params["batch_size"],
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        iterator = iterator.prefetch(tf.data.AUTOTUNE)

        return iterator


class Augmenter:
    def __init__(self, params={}):
        self.params = default_augmenter_params
        self.params.update(params)

        self.normaliser_mean = tf.constant(self.params["normalise_mean"], tf.float32)
        self.normaliser_std = tf.constant(self.params["normalise_std"], tf.float32)

    def __bool__(self):
        return True

    @staticmethod
    def switch_channels(x):
        return tf.transpose(x, [1, 0])

    @tf.function
    def stft(self, x, mode="complex"):
        if x.shape[-1] == 2:  # stereo
            x = self.switch_channels(x)

        complex_x = tf.signal.stft(
            x,
            self.params["fft"]["win"],
            self.params["fft"]["hop"],  # pas temporel de hop/sr
            fft_length=self.params["fft"]["n_fft"],
        )
        if mode == "magnitude":
            return tf.expand_dims(tf.abs(complex_x), -1)

        elif mode == "power":
            return tf.expand_dims(tf.square(tf.abs(complex_x)), -1)

        elif mode == "dB":
            return tf.expand_dims(
                tf.math.log(
                    tf.clip_by_value(
                        tf.square(tf.abs(complex_x)),
                        1e-10,
                        1e6,
                    )
                )
                / tf.math.log(tf.constant(10.0, dtype=tf.float32)),
                -1,
            )

        elif mode == "polar":
            return tf.concat((tf.abs(complex_x), tf.math.angle(complex_x)), -1)

        elif mode == "pure_phase":
            angle = tf.math.angle(complex_x)  # non continu
            print("angle", angle.shape)
            angle_cossin = tf.stack((tf.math.cos(angle), tf.math.sin(angle)), -1)
            return angle_cossin

        else:
            return tf.stack((tf.math.real(complex_x), tf.math.imag(complex_x)), -1)

    @tf.function
    def normaliser(self, x):
        return (x - self.normaliser_mean) / self.normaliser_std

    def random_mono(self, x):
        if x.shape[0] == 2:  # stereo channel first
            x = self.switch_channels(x)

        factor = tf.random.uniform((), minval=0.01, maxval=0.99, dtype=tf.float32)
        return factor * x[:, 0] + (1 - factor) * x[:, 1]

    def random_affine(self, x):
        factor = tf.random.uniform((), minval=0.5, maxval=1.0, dtype=tf.float32)
        return factor * x

    def slice_hf(self, x):
        factor = tf.cast(
            (self.params["hf_cut"] * 2 / self.params["sr"])
            * tf.cast(tf.shape(x)[1], tf.float32),
            tf.int32,
        )
        return tf.slice(x, (0, 0, 0), (-1, factor, -1))

    def slice_lf(
        self,
        x,
    ):
        factor = tf.cast(
            (self.params["lf_cut"] * 2 / self.params["sr"])
            * tf.cast(tf.shape(x)[1], tf.float32),
            tf.int32,
        )
        return tf.slice(x, (0, factor, 0), (-1, -1, -1))

    def add_noise(self, x):
        noise = tf.random.normal(tf.shape(x), stddev=1e-2, dtype=tf.float32)
        return x + noise

    @tf.function
    def transform(self, x, skip_normalise=False):
        y = x

        for effect in self.params["effects"]:
            if effect == "stft_db":
                # import pdb; pdb.set_trace()
                y = self.stft(y, "dB")
            elif effect == "stft_mag":
                y = self.stft(y, "magnitude")
            elif effect == "stft_complex":
                y = self.stft(y, "complex")
            elif effect == "stft_polar":
                y1 = self.stft(y, "pure_phase")
                y2 = self.normaliser(self.stft(y, "dB"))
                y = tf.concat((y1, y2), -1)
            elif effect == "stft_phase":
                y = self.stft(y, "pure_phase")
            elif effect == "normalise" and not skip_normalise:
                y = self.normaliser(y)
            elif effect == "mono":
                y = self.random_mono(y)
            elif effect == "affine":
                y = self.random_affine(y)
            elif effect == "slice_hf":
                y = self.slice_hf(y)
            elif effect == "slice_lf":
                y = self.slice_lf(y)
            elif effect == "noise":
                y = self.add_noise(y)

        return y


class EvalAugmenter(Augmenter):
    def __init__(self, params={}):
        super().__init__(params)

    def random_mono(self, x):
        if x.shape[0] == 2:  # stereo channel first
            x = self.switch_channels(x)
        return 0.5 * x[:, 0] + 0.5 * x[:, 1]

    def random_affine(self, x):
        return x  # disable


class AdversarialAugmenter:
    def __init__(self, effect, params={}):
        self.effects = effect
        if any([effect not in ALL_EFFECTS for effect in self.effects]):
            raise ValueError(
                f"Invalid effect selected: {self.effects}. Choose from {ALL_EFFECTS}"
            )

        self.params = default_adversarial_params
        self.params.update(params)
        self.target_length = int(self.params["target_audio_slice"] * self.params["sr"])

    def create_augmentation(self, audio):
        board = Pedalboard()
        # import pdb; pdb.set_trace()

        for effect in self.effects:
            if effect == "pitch":
                sign = 2 * np.random.randint(0, 2) - 1
                fact = np.random.randint(1, self.params["adversarial_pitch"] + 1)
                semitones = fact * sign
                board.append(PitchShift(semitones=semitones))

            if effect == "reverb":
                board.append(
                    Reverb(
                        room_size=np.random.uniform(0.2, 0.8),
                        damping=np.random.uniform(0.2, 0.8),
                        wet_level=np.random.uniform(0.2, 0.8),
                        dry_level=np.random.uniform(0.2, 0.8),
                        width=np.random.choice([1, np.random.uniform(0.5, 1)]),
                    )
                )
            if effect == "eq":
                # Use HighpassFilter and LowpassFilter for a more controlled EQ effect
                if not random.randrange(3):  # Simulate Bandreject
                    center_freq = np.random.uniform(
                        500, 4000
                    )  # center around 500-4000 Hz
                    band_width = np.random.uniform(
                        50, 500
                    )  # bandwidth between 50-500 Hz

                    highpass_cutoff = center_freq - band_width / 2
                    lowpass_cutoff = center_freq + band_width / 2

                    board.append(HighpassFilter(cutoff_frequency_hz=highpass_cutoff))
                    board.append(LowpassFilter(cutoff_frequency_hz=lowpass_cutoff))
                    board.append(Gain(gain_db=np.random.uniform(-12, -3)))  # gain cut

                if not random.randrange(3):  # Simulate Bass Boost/Cut
                    lowpass_cutoff = np.random.uniform(80, 250)
                    board.append(LowpassFilter(cutoff_frequency_hz=lowpass_cutoff))
                    board.append(Gain(gain_db=np.random.uniform(-12, 12)))

                if not random.randrange(3):  # Simulate Treble Boost/Cut
                    highpass_cutoff = np.random.uniform(2000, 8000)
                    board.append(HighpassFilter(cutoff_frequency_hz=highpass_cutoff))
                    board.append(Gain(gain_db=np.random.uniform(-12, 12)))

            if "noise" in self.effects:
                if np.random.rand() > 0.5:  # Apply noise in 50% of cases
                    noise = np.random.normal(
                        0, self.params["adversarial_noise_std"], (len(audio), 2)
                    )
                    audio += noise

            if "stretch" in self.effects:
                stretch_factor = np.random.uniform(
                    1 - self.params["adversarial_stretch"],
                    1 + self.params["adversarial_stretch"],
                )
                # import pdb; pdb.set_trace()
                # change channels first to channels last
                audio = np.transpose(audio)
                audio = librosa.effects.time_stretch(audio, rate=stretch_factor)
                audio = np.transpose(audio)

        return board, audio

    def py_apply_augmentation(self, audio):
        # Pedalboard expects a NumPy array
        audio_np = audio.numpy()

        board, audio_np = self.create_augmentation(audio_np)

        # Apply the pedalboard
        effected = board(audio_np, sample_rate=self.params["sr"])

        effected = effected / np.max(np.abs(effected))

        return effected

    @tf.function
    def transform(self, x):
        x_shape = tf.shape(x)
        x_ = tf.py_function(self.py_apply_augmentation, [x], tf.float32)
        return tf.slice(x_, (0, 0), x_shape)

    @tf.function
    def batch_transform(self, X):
        X_shape = tf.shape(X)
        X_trick_sox = tf.reshape(tf.transpose(X, [1, 0, 2]), (X_shape[1], -1))
        X_ = tf.py_function(self.py_apply_augmentation, [X_trick_sox], tf.float32)
        X_ = tf.transpose(tf.reshape(X_, (-1, X_shape[0], X_shape[2])), [1, 0, 2])
        # X_ = tf.slice( X_, (0, 0, 0), X_shape)
        return X_
