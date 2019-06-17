import tensorflow as tf
import numpy as np
from text_encoder import JapaneseTextEncoder
import os
import librosa
import copy
import signal
from hyperparams import Hyperparams as hp
from network import encoder, decoder1, decoder2, embed


def make_dataset(path):
    corpus, fpaths = [], []
    with open(path, encoding='utf-8') as f:
        for line in f:
            fname, text = line.strip().split('|')
            fpaths.append(fname)
            corpus.append(text)

    print(corpus)
    encoder = JapaneseTextEncoder(corpus, append_eos=True)
    encoder.build()

    text_lengths = [len(words) for words in encoder.dataset]

    return fpaths, text_lengths, encoder.dataset


def spectrogram2wav(mag):
    '''# Generate wave file from spectrogram'''
    # transpose
    mag = mag.T

    # de-noramlize
    mag = (np.clip(mag, 0, 1) * hp.max_db) - hp.max_db + hp.ref_db

    # to amplitude
    mag = np.power(10.0, mag * 0.05)

    # wav reconstruction
    wav = griffin_lim(mag)

    # de-preemphasis
    wav = signal.lfilter([1], [1, -hp.preemphasis], wav)

    # trim
    wav, _ = librosa.effects.trim(wav)

    return wav.astype(np.float32)


def griffin_lim(spectrogram):
    X_best = copy.deepcopy(spectrogram)
    for i in range(hp.n_iter):
        X_t = invert_spectrogram(X_best)
        est = librosa.stft(X_t, hp.n_fft, hp.hop_length, win_length=hp.win_length)
        phase = est / np.maximum(1e-8, np.abs(est))
        X_best = spectrogram * phase

    X_t = invert_spectrogram(X_best)
    y = np.real(X_t)

    return y


def invert_spectrogram(spectrogram):
    '''
    spectrogram: [f, t]
    '''
    return librosa.istft(spectrogram, hp.hop_length, win_length=hp.win_length, window="hann")


def get_spectrograms(fpath):
    y, sr = librosa.load(fpath, sr=hp.sr)
    y, _ = librosa.effects.trim(y)
    y = np.append(y[0], y[1:] - hp.preemphasis * y[:-1])

    linear = librosa.stft(y=y,
                          n_fft=hp.n_fft,
                          hop_length=hp.hop_length,
                          win_length=hp.win_length)

    mag = np.abs(linear)
    mel_basis = librosa.filters.mel(hp.sr, hp.n_fft, hp.n_mels)
    mel = np.dot(mel_basis, mag)

    # to decibel.
    mel = 20 * np.log10(np.maximum(1e-5, mel))
    mag = 20 * np.log10(np.maximum(1e-5, mag))

    # normalize.
    mel = np.clip((mel - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)
    mag = np.clip((mag - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)

    # transpose.
    mel = mel.T.astype(np.float32)
    mag = mag.T.astype(np.float32)

    return mel, mag


def load_spectrograms(fpath):
    fname = os.path.basename(fpath)
    mel, mag = get_spectrograms(fpath)
    t = mel.shape[0]
    num_paddings = hp.r - (t % hp.r) if t % hp.r != 0 else 0  # for reduction
    mel = np.pad(mel, [[0, num_paddings], [0, 0]], mode="constant")
    mag = np.pad(mag, [[0, num_paddings], [0, 0]], mode="constant")
    return fname, mel.reshape((-1, 80 * 5)), mag


def get_batch():
    with tf.device('/cpu:0'):
        fpaths, text_lengths, dataset = make_dataset('datasets/sentences.dat')
        max_len, min_len = max(text_lengths), min(text_lengths)

        num_batch = len(fpaths) // hp.batch_size
        fpaths = tf.convert_to_tensor(fpaths)
        text_lengths = tf.convert_to_tensor(text_lengths)
        dataset = tf.convert_to_tensor(dataset)

        fpath, text = tf.train.slice_input_producer([fpaths, text_lengths, dataset], shuffle=True)

        fname, mel, mag = tf.py_function(load_spectrograms, [fpath], [tf.string, tf.float32, tf.float32])

        fname.set_shape(())
        text.set_shape((None,))
        mel.set_shape((None, 80 * 5))
        mag.set_shape((None, 2048 // 2 + 1))

        _, (texts, mels, mags, fnames) = tf.contrib.training.bucket_by_sequence_length(
            input_length=50,
            tensors=[text, mel, mag, fname],
            batch_size=hp.batch_size,
            bucket_boundaries=[i for i in range(min_len + 1, max_len - 1, 20)],
            num_threads=16,
            capacity=hp.batch_size * 4,
            dynamic_pad=True
        )

    return texts, mels, mags, fnames, num_batch


class Graph:
    def __init__(self, mode='train'):
        is_training = True if mode == 'train' else False

        if mode == 'train':
            self.x, self.y, self.z, self.fnames, self.num_batch = get_batch()
        elif mode == 'eval':
            self.x = tf.placeholder(tf.int32, shape=(None, None))
            self.y = tf.placeholder(tf.float32, shape=(None, None, hp.n_mels * hp.r))
            self.z = tf.placeholder(tf.float32, shape=(None, None, 1 + hp.n_fft // 2))
            self.fnames = tf.placeholder(tf.string, shape=(None,))
        else:
            # synthesize.
            self.x = tf.placeholder(tf.int32, shape=(None, None))
            self.y = tf.placeholder(tf.float32, shape=(None, None, hp.n_mels * hp.r))

        self.encoder_inputs = embed(self.x, hp.vocab_size, hp.embed_size)
        self.decoder_inputs = tf.concat((tf.zeros_like(self.y[:, :1, :]), self.y[:, :-1, :]), 1)
        self.decoder_inputs = self.decoder_inputs[:, :, -hp.n_mels:]

        # Network.
        with tf.variable_scope("net"):
            self.memory = encoder(self.encoder_inputs, is_training=is_training)
            self.y_hat, self.alignments = decoder1(self.decoder_inputs,
                                                   self.memory,
                                                   is_training=is_training)
            self.z_hat = decoder2(self.y_hat, is_training=is_training)

        self.audio = tf.py_function(spectrogram2wav, [self.z_hat[0]], tf.float32)


if __name__ == '__main__':
    print('Training start.')
