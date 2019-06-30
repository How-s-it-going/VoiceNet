import tensorflow as tf
import numpy as np
from text_encoder import JapaneseTextEncoder
import os
import librosa
import copy
from scipy import signal
from tqdm import tqdm
import matplotlib.pyplot as plt
from hyperparams import Hyperparams as hp
from network import encoder, decoder1, decoder2, embed


def plot_alignment(alignment, gs):
    """Plots the alignment
    alignments: A list of (numpy) matrix of shape (encoder_steps, decoder_steps)
    gs : (int) global step
    """
    fig, ax = plt.subplots()
    im = ax.imshow(alignment)

    # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im)
    plt.title('{} Steps'.format(gs))
    plt.savefig('{}/alignment_{}k.png'.format(hp.logdir, gs // 1000), format='png')


def make_corpus(path):
    corpus, fpaths = [], []
    with open(path, encoding='utf-8') as f:
        for line in f:
            fname, text = line.strip().split('|')
            fpaths.append(fname)
            corpus.append(text)

    encoder = JapaneseTextEncoder(corpus, maxlen=50, padding=True, append_eos=True)
    encoder.build()

    text_lengths = [len(words) for words in encoder.dataset]

    return fpaths, text_lengths, encoder


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
    return fname, mel.reshape((-1, hp.n_mels * hp.r)), mag


def learning_rate_decay(init_lr, global_step, warmup_steps=4000.):
    step = tf.cast(global_step + 1, dtype=tf.float32)
    return init_lr * warmup_steps ** 0.5 * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)


def get_batch():
    with tf.device('/cpu:0'):
        fpaths, text_lengths, encoder = make_corpus('datasets/sentences.dat')

        num_batch = len(fpaths) // hp.batch_size
        fpaths = tf.convert_to_tensor(fpaths)
        text_lengths = tf.convert_to_tensor(text_lengths)
        dataset = tf.convert_to_tensor(encoder.dataset)

        fpath, text_length, text = tf.train.slice_input_producer([fpaths, text_lengths, dataset], shuffle=True)

        fname, mel, mag = tf.py_func(load_spectrograms, [fpath], [tf.string, tf.float32, tf.float32])

        fname.set_shape(())
        text.set_shape((None,))
        mel.set_shape((None, hp.n_mels * hp.r))
        mag.set_shape((None, hp.n_fft // 2 + 1))

        _, (texts, mels, mags, fnames) = tf.contrib.training.bucket_by_sequence_length(
            input_length=text_length,
            tensors=[text, mel, mag, fname],
            batch_size=hp.batch_size,
            bucket_boundaries=[50],
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

        self.audio = tf.py_func(spectrogram2wav, [self.z_hat[0]], tf.float32)

        if mode in ('train', 'eval'):
            # Loss.
            self.loss1 = tf.reduce_mean(tf.abs(self.y_hat - self.y))
            self.loss2 = tf.reduce_mean(tf.abs(self.z_hat - self.z))
            self.loss = self.loss1 + self.loss2

            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.lr = learning_rate_decay(hp.lr, global_step=self.global_step)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)

            self.gvs = self.optimizer.compute_gradients(self.loss)
            self.clipped = []
            for grad, var in self.gvs:
                grad = tf.clip_by_norm(grad, 5.)
                self.clipped.append((grad, var))

            self.training_op = self.optimizer.apply_gradients(self.clipped, global_step=self.global_step)

            tf.summary.scalar('{}/loss1'.format(mode), self.loss1)
            tf.summary.scalar('{}/loss'.format(mode), self.loss)
            tf.summary.scalar('{}/lr'.format(mode), self.lr)

            tf.summary.image('{}/mel_gt'.format(mode), tf.expand_dims(self.y, -1), max_outputs=1)
            tf.summary.image('{}/mel_hat'.format(mode), tf.expand_dims(self.y_hat, -1), max_outputs=1)
            tf.summary.image('{}/mag_gt'.format(mode), tf.expand_dims(self.z, -1), max_outputs=1)
            tf.summary.image('{}/mag_hat'.format(mode), tf.expand_dims(self.z_hat, -1), max_outputs=1)

            tf.summary.audio('{}/sample'.format(mode), tf.expand_dims(self.audio, 0), hp.sr)
            self.merged = tf.summary.merge_all()


if __name__ == '__main__':
    print('Training start.')
    g = Graph()
    sv = tf.train.Supervisor(logdir=hp.logdir, save_summaries_secs=60, save_model_secs=0)

    with sv.managed_session() as sess:
        while 1:
            for _ in tqdm(range(g.num_batch), total=g.num_batch, ncols=70, leave=False, unit='b'):
                _, gs = sess.run([g.training_op, g.global_step])

                if gs % 1000 == 0:
                    sv.saver.save(sess, hp.logdir + '/model_gs_{}k'.format(gs // 1000))

                    al = sess.run(g.alignments)
                    plot_alignment(al[0], gs)

    print('Done')
