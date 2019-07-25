from hyperparams import Hyperparams as hp
import tensorflow as tf


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


def bn(inputs,
       is_training=True,
       activation=None,
       scope='bn',
       reuse=None):
    inputs_shape = inputs.get_shape()
    inputs_rank = inputs_shape.ndims

    with tf.device('/gpu:0'):
        if inputs_rank in [2, 3, 4]:
            if inputs_rank == 2:
                inputs = tf.expand_dims(inputs, axis=1)
                inputs = tf.expand_dims(inputs, axis=2)
            elif inputs_rank == 3:
                inputs = tf.expand_dims(inputs, axis=1)

            outputs = tf.contrib.layers.batch_norm(inputs=inputs,
                                                   center=True,
                                                   scale=True,
                                                   updates_collections=None,
                                                   is_training=is_training,
                                                   scope=scope,
                                                   fused=True,
                                                   reuse=reuse)

            if inputs_rank == 2:
                outputs = tf.squeeze(outputs, axis=[1, 2])
            elif inputs_rank == 3:
                outputs = tf.squeeze(outputs, axis=1)

        else:
            outputs = tf.contrib.layers.batch_norm(inputs=inputs,
                                                   center=True,
                                                   scale=True,
                                                   updates_collections=None,
                                                   is_training=is_training,
                                                   scope=scope,
                                                   fused=False,
                                                   reuse=reuse)

        if activation is not None:
            outputs = activation(outputs)

    return outputs


def conv1d(inputs,
           filters=None,
           size=1,
           rate=1,
           padding="SAME",
           use_bias=False,
           activation=None,
           scope="conv1d",
           reuse=None):
    with tf.variable_scope(scope):
        if padding.lower() == "causal":
            # pre-padding for causality
            pad_len = (size - 1) * rate  # padding size
            inputs = tf.pad(inputs, [[0, 0], [pad_len, 0], [0, 0]])
            padding = "valid"

        if filters is None:
            filters = inputs.get_shape().as_list[-1]

        params = {"inputs": inputs, "filters": filters, "kernel_size": size,
                  "dilation_rate": rate, "padding": padding, "activation": activation,
                  "use_bias": use_bias, "reuse": reuse}

        with tf.device('/gpu:0'):
            outputs = tf.layers.conv1d(**params)
    return outputs


def conv1d_banks(inputs, K=16, is_training=True, scope="conv1d_banks", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        outputs = conv1d(inputs, hp.embed_size // 2, 1)  # k=1
        for k in range(2, K + 1):  # k = 2...K
            with tf.variable_scope("num_{}".format(k)):
                output = conv1d(inputs, hp.embed_size // 2, k)
                outputs = tf.concat((outputs, output), -1)
        outputs = bn(outputs, is_training=is_training, activation=tf.nn.relu)
    return outputs  # (N, T, hp.embed_size//2*K)


def gru(inputs, num_units=None, bidirection=False, scope="gru", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        if num_units is None:
            num_units = inputs.get_shape().as_list[-1]

        cell = DeviceCellWrapper('/gpu:0', tf.contrib.rnn.GRUCell(num_units))
        if bidirection:
            cell_bw = DeviceCellWrapper('/gpu:0', tf.contrib.rnn.GRUCell(num_units))
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell, cell_bw, inputs, dtype=tf.float32)
            return tf.concat(outputs, 2)
        else:
            outputs, _ = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
            return outputs


def attention_decoder(inputs, memory, num_units=None, scope="attention_decoder", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        if num_units is None:
            num_units = inputs.get_shape().as_list[-1]

        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units,
                                                                   memory)
        decoder_cell = DeviceCellWrapper('/gpu:0', tf.contrib.rnn.GRUCell(num_units))
        cell_with_attention = tf.contrib.seq2seq.AttentionWrapper(decoder_cell,
                                                                  attention_mechanism,
                                                                  num_units,
                                                                  alignment_history=True)
        outputs, state = tf.nn.dynamic_rnn(cell_with_attention, inputs, dtype=tf.float32)  # ( N, T', 16)

    return outputs, state


def prenet(inputs, num_units=None, is_training=True, scope="prenet", reuse=None):
    if num_units is None:
        num_units = [hp.embed_size, hp.embed_size // 2]

    with tf.variable_scope(scope, reuse=reuse):
        with tf.device('/gpu:0'):
            outputs = tf.layers.dense(inputs, units=num_units[0], activation=tf.nn.relu, name="dense1")
            outputs = tf.layers.dropout(outputs, rate=hp.dropout_rate, training=is_training, name="dropout1")
            outputs = tf.layers.dense(outputs, units=num_units[1], activation=tf.nn.relu, name="dense2")
            outputs = tf.layers.dropout(outputs, rate=hp.dropout_rate, training=is_training, name="dropout2")
    return outputs  # (N, ..., num_units[1])


def highwaynet(inputs, num_units=None, scope="highwaynet", reuse=None):
    if not num_units:
        num_units = inputs.get_shape()[-1]

    with tf.variable_scope(scope, reuse=reuse):
        with tf.device('/gpu:0'):
            H = tf.layers.dense(inputs, units=num_units, activation=tf.nn.relu, name="dense1")
            T = tf.layers.dense(inputs, units=num_units, activation=tf.nn.sigmoid,
                                bias_initializer=tf.constant_initializer(-1.0), name="dense2")
            outputs = H * T + inputs * (1. - T)
    return outputs


# encoder/decoder networks.
def encoder(inputs, is_training=True, scope="encoder", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        # Encoder pre-net
        prenet_out = prenet(inputs, is_training=is_training)  # (N, T_x, E/2)

        # Encoder CBHG
        # Conv1D banks
        enc = conv1d_banks(prenet_out, K=hp.encoder_num_banks, is_training=is_training)  # (N, T_x, K*E/2)

        with tf.device('/gpu:0'):
            # Max pooling
            enc = tf.layers.max_pooling1d(enc, pool_size=2, strides=1, padding="same")  # (N, T_x, K*E/2)

        # Conv1D projections
        enc = conv1d(enc, filters=hp.embed_size // 2, size=3, scope="conv1d_1")  # (N, T_x, E/2)
        enc = bn(enc, is_training=is_training, activation=tf.nn.relu, scope="conv1d_1")

        enc = conv1d(enc, filters=hp.embed_size // 2, size=3, scope="conv1d_2")  # (N, T_x, E/2)
        enc = bn(enc, is_training=is_training, scope="conv1d_2")

        enc += prenet_out  # (N, T_x, E/2) # residual connections

        # Highway Nets
        for i in range(hp.num_highwaynet_blocks):
            enc = highwaynet(enc, num_units=hp.embed_size // 2,
                             scope='highwaynet_{}'.format(i))  # (N, T_x, E/2)

        # Bidirectional GRU
        memory = gru(enc, num_units=hp.embed_size // 2, bidirection=True)  # (N, T_x, E)

    return memory


def decoder1(inputs, memory, is_training=True, scope="decoder1", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        # Decoder pre-net
        inputs = prenet(inputs, is_training=is_training)  # (N, T_y/r, E/2)

        # Attention RNN
        dec, state = attention_decoder(inputs, memory, num_units=hp.embed_size)  # (N, T_y/r, E)

        # for attention monitoring
        alignments = tf.transpose(state.alignment_history.stack(), [1, 2, 0])

        # Decoder RNNs
        dec += gru(dec, hp.embed_size, bidirection=False, scope="decoder_gru1")  # (N, T_y/r, E)
        dec += gru(dec, hp.embed_size, bidirection=False, scope="decoder_gru2")  # (N, T_y/r, E)

        with tf.device('/gpu:0'):
            # Outputs => (N, T_y/r, n_mels*r)
            mel_hats = tf.layers.dense(dec, hp.n_mels * hp.r)

    return mel_hats, alignments


def decoder2(inputs, is_training=True, scope="decoder2", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        with tf.device('/gpu:0'):
            # Restore shape -> (N, Ty, n_mels)
            inputs = tf.reshape(inputs, [tf.shape(inputs)[0], -1, hp.n_mels])

        # Conv1D bank
        dec = conv1d_banks(inputs, K=hp.decoder_num_banks, is_training=is_training)  # (N, T_y, E*K/2)

        with tf.device('/gpu:0'):
            # Max pooling
            dec = tf.layers.max_pooling1d(dec, pool_size=2, strides=1, padding="same")  # (N, T_y, E*K/2)

        # Conv1D projections
        dec = conv1d(dec, filters=hp.embed_size // 2, size=3, scope="conv1d_1")  # (N, T_x, E/2)
        dec = bn(dec, is_training=is_training, activation=tf.nn.relu, scope="conv1d_1")

        dec = conv1d(dec, filters=hp.n_mels, size=3, scope="conv1d_2")  # (N, T_x, E/2)
        dec = bn(dec, is_training=is_training, scope="conv1d_2")

        with tf.device('/gpu:0'):
            # Extra affine transformation for dimensionality sync
            dec = tf.layers.dense(dec, hp.embed_size // 2)  # (N, T_y, E/2)

        # Highway Nets
        for i in range(4):
            dec = highwaynet(dec, num_units=hp.embed_size // 2,
                             scope='highwaynet_{}'.format(i))  # (N, T_y, E/2)

        # Bidirectional GRU
        dec = gru(dec, hp.embed_size // 2, bidirection=True)  # (N, T_y, E)

        with tf.device('/gpu:0'):
            # Outputs => (N, T_y, 1+n_fft//2)
            outputs = tf.layers.dense(dec, 1 + hp.n_fft // 2)

    return outputs


class DeviceCellWrapper(tf.nn.rnn_cell.RNNCell):
    def __init__(self, device, cell):
        self._cell = cell
        self._device = device

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def __call__(self, inputs, state, scope=None):
        with tf.device(self._device):
            return self._cell(inputs, state, scope)
