import tensorflow as tf
import numpy as np
import MeCab
from text_encoder import JapaneseTextEncoder
import os, sys, codecs
import librosa


def make_dataset(path):
    corpus, fpaths = [], []
    with open(path, encoding='utf-8') as f:
        for line in f:
            fname, text = line.strip().split('|')
            fpaths.append(fname)
            corpus.append(text)

    print(corpus)
    encoder = JapaneseTextEncoder(corpus, append_eos=True, maxlen=50, padding=True)
    encoder.build()

    return fpaths, encoder.dataset


def get_spectrograms(fpath):
    y, sr = librosa.load(fpath, sr=22050)
    y, _ = librosa.effects.trim(y)
    y = np.append(y[0], y[1:] - 0.97 * y[:-1])

    linear = librosa.stft(y=y,
                          n_fft=2048,
                          hop_length=int(sr * 0.0125),
                          win_length=int(sr * 0.05))

    mag = np.abs(linear)
    mel_basis = librosa.filters.mel(sr, 2048, 80)
    mel = np.dot(mel_basis, mag)

    # to decibel.
    mel = 20 * np.log10(np.maximum(1e-5, mel))
    mag = 20 * np.log10(np.maximum(1e-5, mag))

    # normalize.
    mel = np.clip((mel - 20 + 100) / 100, 1e-8, 1)
    mag = np.clip((mag - 20 + 100) / 100, 1e-8, 1)

    # transpose.
    mel = mel.T.astype(np.float32)
    mag = mag.T.astype(np.float32)

    return mel, mag


def load_spectrograms(fpath):
    fname = os.path.basename(fpath)
    mel, mag = get_spectrograms(fpath)
    t = mel.shape[0]
    num_paddings = 5 - (t % 5) if t % 5 != 0 else 0  # for reduction
    mel = np.pad(mel, [[0, num_paddings], [0, 0]], mode="constant")
    mag = np.pad(mag, [[0, num_paddings], [0, 0]], mode="constant")
    return fname, mel.reshape((-1, 80 * 5)), mag


def embed(inputs, vocab_size, num_units, zero_pad=True, scope='embedding', reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        lookup_table = tf.get_variable('lookup_table',
                                       dtype=tf.float32,
                                       shape=[vocab_size, num_units],
                                       initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))

        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)

    return tf.nn.embedding_lookup(lookup_table, inputs)


def get_batch(batch_size=1):
    with tf.device('/cpu:0'):
        fpaths, dataset = make_dataset('datasets/sentences.dat')
        num_batch = len(fpaths) // batch_size
        fpaths = tf.convert_to_tensor(fpaths)
        dataset = tf.convert_to_tensor(dataset)

        fpath, text = tf.train.slice_input_producer([fpaths, dataset], shuffle=True)

        fname, mel, mag = tf.py_function(load_spectrograms, [fpath], [tf.string, tf.float32, tf.float32])

        fname.set_shape(())
        text.set_shape((None,))
        mel.set_shape((None, 80 * 5))
        mag.set_shape((None, 2048 // 2 + 1))

        _, (texts, mels, mags, fnames) = tf.contrib.training.bucket_by_sequence_length(
            input_length=50,
            tensors=[text, mel, mag, fname],
            batch_size=batch_size,
            num_threads=16,
            capacity=batch_size * 4,
            dynamic_pad=True
        )

    return texts, mels, mags, fnames, num_batch


class Graph:
    def __init__(self, mode='train'):
        is_training = True if mode == 'train' else False

        if mode == 'train':
            self.x, self.y, self.z, self.fnames, self.num_batch = get_batch(16)
        elif mode == 'eval':
            self.x = tf.placeholder(tf.int32, shape=(None, None))
            self.y = tf.placeholder(tf.float32, shape=(None, None, 80 * 5))
            self.z = tf.placeholder(tf.float32, shape=(None, None, 1 + 2048 // 2))
            self.fnames = tf.placeholder(tf.string, shape=(None,))
        else:
            self.x = tf.placeholder(tf.int32, shape=(None, None))
            self.y = tf.placeholder(tf.float32, shape=(None, None, 80 * 5))


if __name__ == '__main__':
    print('Training start.')
