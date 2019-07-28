import tensorflow as tf


class Hyperparams:
    # signal processing.
    vocab_size = 400000
    sr = 22050
    n_fft = 2048  # fft points (samples)
    frame_shift = 0.0125  # seconds
    frame_length = 0.05  # seconds
    hop_length = int(sr * frame_shift)  # samples.
    win_length = int(sr * frame_length)  # samples.
    n_mels = 80  # Number of Mel banks to generate
    power = 1.2  # Exponent for amplifying the predicted magnitude
    n_iter = 50  # Number of inversion iterations
    preemphasis = .97  # or None
    max_db = 100
    ref_db = 20

    # model
    embed_size = 256  # alias = E
    encoder_num_banks = 16
    decoder_num_banks = 8
    num_highwaynet_blocks = 4
    r = 5  # Reduction factor. Paper => 2, 3, 5
    dropout_rate = .5

    # training scheme
    lr = 0.001  # Initial learning rate.
    logdir = "logdir/01"
    sampledir = 'samples'
    batch_size = 32

    cluster_spec = tf.train.ClusterSpec({
        'ps': [
            '192.168.1.12:2221',  # /job:ps/task:0
            '192.168.1.10:2221'  # /job:ps/task:1
        ],
        'worker': [
            '10.123.123.123:2222'  # /job:worker/task:0 (Docker Image)
        ]
    })
