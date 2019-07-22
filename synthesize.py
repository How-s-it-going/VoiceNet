from hyperparams import Hyperparams as hp
import numpy as np
import tensorflow as tf
from scipy.io.wavfile import write
import os
from model import Graph, spectrogram2wav, make_corpus
from tqdm import tqdm


def synthesize():
    if not os.path.exists(hp.sampledir): os.mkdir(hp.sampledir)
    # Load graph
    g = Graph(mode="synthesize")
    print("Graph loaded")

    _, _, encoder = make_corpus('datasets/sentences.dat')
    sent = input('Input a sentence.')
    text = np.array([encoder.encode(sent)], dtype=np.int32)
    print(text)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(hp.logdir))
        print('Restored !!')

        # Feed Forward
        ## mel
        y_hat = np.zeros((text.shape[0], 200, hp.n_mels * hp.r), np.float32)  # hp.n_mels*hp.r
        for j in tqdm(range(200)):
            _y_hat = sess.run(g.y_hat, {g.x: text, g.y: y_hat})
            y_hat[:, j, :] = _y_hat[:, j, :]
        ## mag
        mags = sess.run(g.z_hat, {g.y_hat: y_hat})
        for i, mag in enumerate(mags):
            print("File {}.wav is being generated ...".format(i + 1))
            audio = spectrogram2wav(mag)
            write(os.path.join(hp.sampledir, '{}.wav'.format(i + 1)), hp.sr, audio)


if __name__ == '__main__':
    synthesize()
    print("Done")
