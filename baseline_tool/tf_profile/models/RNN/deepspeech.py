
from six.moves import xrange
import tensorflow as tf
import horovod.tensorflow as hvd
tf.set_random_seed(1)

# Supported rnn cells.
SUPPORTED_RNNS = {
    "lstm": tf.nn.rnn_cell.BasicLSTMCell,
    "rnn": tf.nn.rnn_cell.RNNCell,
    "gru": tf.nn.rnn_cell.GRUCell,
}

# Parameters for batch normalization.
_BATCH_NORM_EPSILON = 1e-5
_BATCH_NORM_DECAY = 0.997

# Filters of convolution layer
_CONV_FILTERS = 32


def batch_norm(inputs, training):
    """Batch normalization layer.

    Note that the momentum to use will affect validation accuracy over time.
    Batch norm has different behaviors during training/evaluation. With a large
    momentum, the model takes longer to get a near-accurate estimation of the
    moving mean/variance over the entire training dataset, which means we need
    more iterations to see good evaluation results.
    If the training data is evenly
    distributed over the feature space,
    we can also try setting a smaller momentum
    (such as 0.1) to get good evaluation result sooner.

    Args:
      inputs: input data for batch norm layer.
      training: a boolean to indicate if it is in training stage.

    Returns:
      tensor output from batch norm layer.
    """
    # return inputs
    # print(inputs)
    return tf.layers.batch_normalization(
        inputs=inputs, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON,
        fused=True, training=training)


def _conv_bn_layer(inputs, padding, filters, kernel_size, strides, layer_id,
                   training):
    """Defines 2D convolutional + batch normalization layer.

    Args:
      inputs: input data for convolution layer.
      padding: padding to be applied before convolution layer.
      filters: an integer, number of output filters in the convolution.
      kernel_size: a tuple specifying the height
      and width of the 2D convolution
        window.
      strides: a tuple specifying the stride length of the convolution.
      layer_id: an integer specifying the layer index.
      training: a boolean to indicate which stage we are in (training/eval).

    Returns:
      tensor output from the current layer.
    """
    # Perform symmetric padding on the feature dimension of time_step
    # This step is required to avoid issues when RNN output sequence is shorter
    # than the label length.
    inputs = tf.pad(
        inputs,
        [[0, 0], [padding[0], padding[0]], [padding[1], padding[1]], [0, 0]])
    inputs = tf.layers.conv2d(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding="valid",
        use_bias=False,
        activation=tf.nn.relu6,
        name="cnn_{}".format(layer_id))
    return batch_norm(inputs, training)


def _rnn_layer(
        inputs,
        rnn_cell,
        rnn_hidden_size,
        seq_length,
        layer_id,
        is_batch_norm,
        is_bidirectional,
        training):
    """Defines a batch normalization + rnn layer.

    Args:
      inputs: input tensors for the current layer.
      rnn_cell: RNN cell instance to use.
      rnn_hidden_size: an integer for the dimensionality
      of the rnn output space.
      layer_id: an integer for the index of current layer.
      is_batch_norm: a boolean specifying whether to perform
      batch normalization
        on input states.
      is_bidirectional: a boolean specifying whether the rnn layer is
        bi-directional.
      training: a boolean to indicate which stage we are in (training/eval).

    Returns:
      tensor output for the current layer.
    """
    # rnn_length=len(inputs)
    # if is_batch_norm:
    #  inputs = tf.concat(inputs,axis=0)
    #  inputs = batch_norm(inputs, training)
    #  inputs = tf.split(value=inputs, axis=0, num_or_size_splits=rnn_length)

    # print(inputs.get_shape())
    # print("!!!")

    # Construct forward/backward RNN cells.
    fw_cell = rnn_cell(num_units=rnn_hidden_size,
                       name="rnn_fw_{}".format(layer_id))
    bw_cell = rnn_cell(num_units=rnn_hidden_size,
                       name="rnn_bw_{}".format(layer_id))

    # inputs = tf.split(value=inputs, axis=1, num_or_size_splits=seq_length)
    # for i in range(len(inputs)):
    #  inputs[i] = tf.squeeze(inputs[i],axis=[1])

    if is_bidirectional:
        outputs, _fw, _bw = tf.nn.static_bidirectional_rnn(
            cell_fw=fw_cell, cell_bw=bw_cell, inputs=inputs, dtype=tf.float32)
        # rnn_outputs = tf.concat(outputs, -1)
        rnn_outputs = outputs
    else:
        rnn_outputs, _ = tf.nn.static_rnn(
            fw_cell, inputs, dtype=tf.float32)

    return rnn_outputs


class DeepSpeech2(object):
    """Define DeepSpeech2 model."""

    def __init__(self, num_rnn_layers, rnn_type, is_bidirectional,
                 rnn_hidden_size, num_classes, use_bias):
        """Initialize DeepSpeech2 model.

        Args:
          num_rnn_layers: an integer, the number of rnn layers.
          By default, it's 5.
          rnn_type: a string, one of the supported rnn cells:
          gru, rnn and lstm.
          is_bidirectional: a boolean to indicate if the rnn
          layer is bidirectional.
          rnn_hidden_size: an integer for the number of
          hidden states in each unit.
          num_classes: an integer, the number of output classes/labels.
          use_bias: a boolean specifying whether to use
          bias in the last fc layer.
        """
        self.num_rnn_layers = num_rnn_layers
        self.rnn_type = rnn_type
        self.is_bidirectional = is_bidirectional
        self.rnn_hidden_size = rnn_hidden_size
        self.num_classes = num_classes
        self.use_bias = use_bias

    def __call__(self, inputs, training):
        # Two cnn layers.
        inputs = _conv_bn_layer(
            inputs, padding=(
                20, 5), filters=_CONV_FILTERS, kernel_size=(
                41, 11), strides=(
                2, 2), layer_id=1, training=training)

        inputs = _conv_bn_layer(
            inputs, padding=(
                10, 5), filters=_CONV_FILTERS, kernel_size=(
                21, 11), strides=(
                2, 1), layer_id=2, training=training)

        # output of conv_layer2 with the shape of
        # [batch_size (N), times (T), features (F), channels (C)].
        # Convert the conv output to rnn input.
        batch_size = tf.shape(inputs)[0]
        feat_size = inputs.get_shape().as_list()[2]
        # print(inputs)
        inputs = tf.reshape(
            inputs,
            [batch_size, -1, feat_size * _CONV_FILTERS])

        rnn_length = inputs.get_shape().as_list()[1]
        rnn_length = 75
        inputs = tf.split(value=inputs, axis=1, num_or_size_splits=rnn_length)
        for i in range(len(inputs)):
            inputs[i] = tf.squeeze(inputs[i], axis=[1])

        # RNN layers.
        rnn_cell = SUPPORTED_RNNS[self.rnn_type]
        for layer_counter in xrange(self.num_rnn_layers):
            # No batch normalization on the first layer.
            is_batch_norm = (layer_counter != 0)
            # is_batch_norm = False
            inputs = _rnn_layer(
                inputs,
                rnn_cell,
                self.rnn_hidden_size,
                rnn_length,
                layer_counter + 1,
                is_batch_norm,
                self.is_bidirectional,
                training)

        inputs = tf.concat(inputs, axis=0)

        # # FC layer with batch norm.
        inputs = batch_norm(inputs, training)
        # print(inputs.get_shape())
        logits = tf.layers.dense(
            inputs,
            self.num_classes,
            use_bias=self.use_bias)

        # print(logits.get_shape())

        # logits = inputs

        return logits


def DeepSpeech2Fn(batchsize=None):
    batch_size = 16
    # use specified batch_size if not None
    if batchsize is not None:
        batch_size = batchsize

    num_classes = 29
    height = 300  # voice length
    hidden_size = 256

    # eval_inputs = tf.placeholder(
    #     tf.float32, [batch_size, height, 171, 1], 'eval_input')
    # image_inputs = np.ones((batch_size, height, 171, 1))
    eval_inputs = tf.random_uniform((batch_size, height, 171, 1))

    labels = tf.zeros(shape=[batch_size], dtype=tf.float32, name="labels")
    labels = tf.cast(labels, tf.int64, name="labels_cast")

    model = DeepSpeech2(
        num_rnn_layers=7,
        rnn_type='lstm',
        is_bidirectional=False,
        rnn_hidden_size=hidden_size,
        num_classes=num_classes,
        use_bias=True)

    logits = model(eval_inputs, True)
    probs = tf.nn.softmax(logits)
    loss = tf.reduce_mean(probs)

    opt = tf.train.GradientDescentOptimizer(0.123456)  # lr
    grads = opt.compute_gradients(loss)
    apply_gradient_op = opt.apply_gradients(grads)
    return apply_gradient_op, loss


def DeepSpeech2Fn_horovod(batchsize=None):
    batch_size = 16
    # use specified batch_size if not None
    if batchsize is not None:
        batch_size = batchsize

    num_classes = 29
    height = 300  # voice length
    hidden_size = 256

    # eval_inputs = tf.placeholder(
    #     tf.float32, [batch_size, height, 171, 1], 'eval_input')
    # image_inputs = np.ones((batch_size, height, 171, 1))
    eval_inputs = tf.random_uniform((batch_size, height, 171, 1))

    labels = tf.zeros(shape=[batch_size], dtype=tf.float32, name="labels")
    labels = tf.cast(labels, tf.int64, name="labels_cast")

    model = DeepSpeech2(
        num_rnn_layers=7,
        rnn_type='lstm',
        is_bidirectional=False,
        rnn_hidden_size=hidden_size,
        num_classes=num_classes,
        use_bias=True)

    logits = model(eval_inputs, True)
    probs = tf.nn.softmax(logits)
    loss = tf.reduce_mean(probs)

    # Initialize Horovod
    hvd.init()
    print("hvd.size(): %d" % hvd.size())

    opt = tf.train.GradientDescentOptimizer(0.123456 * hvd.size())  # lr
    opt = hvd.DistributedOptimizer(opt)

    grads = opt.compute_gradients(loss)
    apply_gradient_op = opt.apply_gradients(grads)
    return apply_gradient_op, loss
