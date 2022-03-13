import tensorflow as tf
import horovod.tensorflow as hvd
tf.set_random_seed(1)


class ExtendedMultiRnnCell(object):

    def __init__(self, cells):
        self.cells = cells

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or "extended_multi_rnn_cell"):
            cur_input = inputs
            # prev_inputs = [cur_input]
            new_states = []
            for i in range(len(self.cells)):
                cell = self.cells[i]
                cur_state = state[i]
                next_input, new_state = cell(cur_input, cur_state)
                cur_input = next_input
                new_states.append(new_state)
        new_states = tuple(new_states)
        return cur_input, new_states


class Seq2SeqModel(object):

    def __init__(
            self,
            batch_size,
            hidden_size,
            num_encoder_layer,
            encoder_step,
            num_decoder_layer,
            decoder_step):
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_encoder_layer = num_encoder_layer
        self.num_decoder_layer = num_decoder_layer
        self.encoder_step = encoder_step
        self.decoder_step = decoder_step

    def _rnn_layer(
            self,
            inputs,
            rnn_hidden_size,
            seq_length,
            layer_id,
            is_bidirectional=False):
        """Defines a batch normalization + rnn layer.

        Args:
            inputs: input tensors for the current layer.
            rnn_cell: RNN cell instance to use.
            rnn_hidden_size: an integer for the dimensionality
            of the rnn output space.
            layer_id: an integer for the index of current layer.
            is_bidirectional: a boolean specifying whether the rnn layer is
            bi-directional.

        Returns:
            tensor output for the current layer.
        """
        rnn_cell = tf.nn.rnn_cell.BasicLSTMCell
        # rnn_length=len(inputs)
        # if is_batch_norm:
        #   inputs = tf.concat(inputs,axis=0)
        #   inputs = batch_norm(inputs, training)
        #   inputs = tf.split(value=inputs, axis=0,
        # num_or_size_splits=rnn_length)

        # print(inputs.get_shape())
        # print("!!!")

        # Construct forward/backward RNN cells.
        fw_cell = rnn_cell(num_units=rnn_hidden_size,
                           name="encoder_rnn_fw_{}".format(layer_id))
        bw_cell = rnn_cell(num_units=rnn_hidden_size,
                           name="encoder_rnn_bw_{}".format(layer_id))

        # inputs = tf.split(value=inputs, axis=1,
        # num_or_size_splits=seq_length)
        # for i in range(len(inputs)):
        #   inputs[i] = tf.squeeze(inputs[i],axis=[1])

        if is_bidirectional:
            outputs, _fw, _bw = tf.nn.static_bidirectional_rnn(
                cell_fw=fw_cell, cell_bw=bw_cell, inputs=inputs,
                dtype=tf.float32)
            # rnn_outputs = tf.concat(outputs, -1)
            _ = _fw
            rnn_outputs = outputs
        else:
            rnn_outputs, _ = tf.nn.static_rnn(
                fw_cell, inputs, dtype=tf.float32)

        return rnn_outputs, _

    def _build_encoder(self, inputs):
        for layer_counter in range(self.num_encoder_layer):
            inputs, encoder_state = self._rnn_layer(
                inputs, self.hidden_size, self.encoder_step,
                layer_counter + 1, is_bidirectional=False)

        return encoder_state

    def _build_decoder(self, inputs):
        rnn_cell = tf.nn.rnn_cell.BasicLSTMCell
        cells = []
        for layer_counter in range(self.num_decoder_layer):
            cell = rnn_cell(num_units=self.hidden_size,
                            name="decoder_rnn_fw_{}".format(layer_counter))
            cells.append(cell)
        if len(cells) > 1:
            final_cell = ExtendedMultiRnnCell(cells)
        else:
            final_cell = cells[0]

        cur_states = []
        for i in range(len(cells)):
            cur_states.append(
                cells[i].zero_state(
                    self.batch_size,
                    dtype=tf.float32))
        if len(cur_states) == 1:
            cur_states = cur_states[0]

        cur_input = inputs[0]

        for step in range(self.decoder_step):
            next_input, next_states = final_cell(cur_input, cur_states)
            cur_input = next_input
            cur_states = next_states

        return cur_input

    def __call__(self, inputs):
        encoder_state = self._build_encoder(inputs)
        decoder_state = self._build_decoder(encoder_state)

        return decoder_state


FLAGS = {
    "encoder_step": 100,
    "encoder_layer": 8,
    "decoder_step": 30,
    "decoder_layer": 4,
    "hidden_size": 128,
    "batch_size": 16,
}


def Seq2seqFn(batchsize=None):
    # use specified batch_size if not None
    if batchsize is not None:
        FLAGS["batch_size"] = batchsize

    model = Seq2SeqModel(
        FLAGS["batch_size"],
        FLAGS["hidden_size"],
        FLAGS["encoder_layer"],
        FLAGS["encoder_step"],
        FLAGS["decoder_layer"],
        FLAGS["decoder_step"])

    eval_inputs = tf.random.uniform((FLAGS["encoder_step"],
                                     FLAGS["batch_size"],
                                     FLAGS["hidden_size"]))
    eval_inputs_list = tf.split(
        value=eval_inputs,
        axis=0,
        num_or_size_splits=FLAGS["encoder_step"])
    for i in range(len(eval_inputs_list)):
        eval_inputs_list[i] = tf.squeeze(eval_inputs_list[i], axis=[0])

    logits = model(eval_inputs_list)

    target = tf.random.uniform((FLAGS["batch_size"], FLAGS["hidden_size"]))
    loss = tf.reduce_mean(tf.square(logits-target))

    opt = tf.train.GradientDescentOptimizer(0.123456)  # lr
    grads = opt.compute_gradients(loss)
    apply_gradient_op = opt.apply_gradients(grads)
    return apply_gradient_op, loss


def Seq2seqFn_horovod(batchsize=None):
    # use specified batch_size if not None
    if batchsize is not None:
        FLAGS["batch_size"] = batchsize

    model = Seq2SeqModel(
        FLAGS["batch_size"],
        FLAGS["hidden_size"],
        FLAGS["encoder_layer"],
        FLAGS["encoder_step"],
        FLAGS["decoder_layer"],
        FLAGS["decoder_step"])

    eval_inputs = tf.random.uniform((FLAGS["encoder_step"],
                                     FLAGS["batch_size"],
                                     FLAGS["hidden_size"]))
    eval_inputs_list = tf.split(
        value=eval_inputs,
        axis=0,
        num_or_size_splits=FLAGS["encoder_step"])
    for i in range(len(eval_inputs_list)):
        eval_inputs_list[i] = tf.squeeze(eval_inputs_list[i], axis=[0])

    logits = model(eval_inputs_list)

    target = tf.random.uniform((FLAGS["batch_size"], FLAGS["hidden_size"]))
    loss = tf.reduce_mean(tf.square(logits-target))

    # Initialize Horovod
    hvd.init()
    print("hvd.size(): %d" % hvd.size())

    opt = tf.train.GradientDescentOptimizer(0.123456 * hvd.size())  # lr
    opt = hvd.DistributedOptimizer(opt)

    grads = opt.compute_gradients(loss)
    apply_gradient_op = opt.apply_gradients(grads)
    return apply_gradient_op, loss
