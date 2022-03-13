import tensorflow as tf
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
import horovod.tensorflow as hvd
tf.set_random_seed(1)

FLAGS = {
    "num_step": 100,
    "num_layer": 10,
    "hidden_size": 256,
    "batch_size": 16,
}


class LSTMCell(object):
    W = []
    U = []
    b = []

    def __init__(self, hidden_size, scope):
        with tf.variable_scope(scope):
            self.W = []
            self.U = []
            self.b = []
            self.num_unit = hidden_size
            for i in range(4):
                W = tf.get_variable(
                    "W%d" %
                    (i), [
                        self.num_unit, self.num_unit], dtype=tf.float32)
                U = tf.get_variable(
                    "U%d" %
                    (i), [
                        self.num_unit, self.num_unit], dtype=tf.float32)
                b = tf.get_variable(
                    "bias%d" %
                    (i), [
                        self.num_unit], dtype=tf.float32,
                    initializer=init_ops.constant_initializer(
                        0, dtype=tf.float32))
                self.W.append(W)
                self.U.append(U)
                self.b.append(b)

    def call(self, inputs, state):
        c, h = state
        res = []
        for i in range(4):
            res.append(math_ops.matmul(
                inputs, self.W[i]) + math_ops.matmul(h, self.U[i]) + self.b[i])
        i, j, f, o = (res[0], res[1], res[2], res[3])
        new_c = (c * math_ops.sigmoid(f + 1.0) +
                 math_ops.sigmoid(i) * math_ops.tanh(j))
        new_h = math_ops.tanh(new_c) * math_ops.sigmoid(o)
        new_state = tf.nn.rnn_cell.LSTMStateTuple(new_c, new_h)
        return new_h, new_state


class LSTMModel(object):
    stacked_cells = []

    def __init__(self, num_layer, hidden_size):
        self.stacked_cells = []
        self.num_layer = num_layer
        self.num_unit = hidden_size
        for layer in range(self.num_layer):
            self.stacked_cells.append(
                LSTMCell(self.num_unit,  "LSTMLayer%d" % (layer)))

    def run(self, inputs, batch_size, num_step):
        self.batch_size = batch_size
        self.num_step = num_step

        cell = tf.nn.rnn_cell.BasicLSTMCell(
            self.num_unit, forget_bias=1.0, state_is_tuple=True)
        self._initial_state = cell.zero_state(batch_size, tf.float32)
        self.state = [self._initial_state for layer in range(self.num_layer)]

        for step in range(self.num_step):
            cur_input = inputs[step, :, :]
            for layer in range(self.num_layer):
                cell_output, self.state[layer] = \
                    self.stacked_cells[layer].call(
                    cur_input, self.state[layer])
                cur_input = cell_output

        self.output = cell_output
        return self.output, self.state[-1]


def tf_lstm_model(inputs):
    def lstm_cell():
        return tf.nn.rnn_cell.LSTMCell(FLAGS["hidden_size"])
    stacked_cell = tf.nn.rnn_cell.MultiRNNCell(
        [lstm_cell() for i in range(FLAGS["num_layer"])])

    state = stacked_cell.zero_state(
        FLAGS["batch_size"], tf.float32)

    for step in range(FLAGS["num_step"]):
        output, state = stacked_cell(inputs[step, :, :], state)

    return output, state


def LstmFn(batchsize=None):
    # use specified batch_size if not None
    if batchsize is not None:
        FLAGS["batch_size"] = batchsize

    model = LSTMModel(FLAGS["num_layer"], FLAGS["hidden_size"])
    lstm_inputs = tf.random.uniform((FLAGS["num_step"],
                                     FLAGS["batch_size"],
                                     FLAGS["hidden_size"]))
    target = tf.random.uniform((FLAGS["batch_size"], FLAGS["hidden_size"]))

    lstm_output, lstm_state = model.run(
        lstm_inputs, FLAGS["batch_size"], FLAGS["num_step"])
    loss = tf.reduce_mean(tf.square(lstm_output-target))

    opt = tf.train.GradientDescentOptimizer(0.123456)  # lr
    grads = opt.compute_gradients(loss)
    apply_gradient_op = opt.apply_gradients(grads)
    return apply_gradient_op, loss


def LstmFn_horovod(batchsize=None):
    # use specified batch_size if not None
    if batchsize is not None:
        FLAGS["batch_size"] = batchsize

    model = LSTMModel(FLAGS["num_layer"], FLAGS["hidden_size"])
    lstm_inputs = tf.random.uniform((FLAGS["num_step"], FLAGS["batch_size"],
                                     FLAGS["hidden_size"]))
    target = tf.random.uniform((FLAGS["batch_size"], FLAGS["hidden_size"]))

    lstm_output, lstm_state = model.run(
        lstm_inputs, FLAGS["batch_size"], FLAGS["num_step"])

    # Initialize Horovod
    hvd.init()
    print("hvd.size(): %d" % hvd.size())

    # Build model...
    loss = tf.reduce_mean(tf.square(lstm_output-target))
    opt = tf.train.GradientDescentOptimizer(0.123456 * hvd.size())  # lr

    opt = hvd.DistributedOptimizer(opt)

    grads = opt.compute_gradients(loss)
    apply_gradient_op = opt.apply_gradients(grads)
    return apply_gradient_op, loss
