import tensorflow as tf
import horovod.tensorflow as hvd
from . import resnet
from . import vgg
from . import alexnet
from . import convnet_builder
from . import inception


_model_name_to_imagenet_model = {
    'vgg16': vgg.Vgg16Model,
    'inception3': inception.Inceptionv3Model,
    'resnet50': resnet.create_dlbs_resnet50,
}

_model_name_to_cifar_model = {
    'alexnet': alexnet.AlexnetCifar10Model,
}


class Dataset():
    def __init__(self, dataset_name):
        if dataset_name not in ["imagenet", "cifar10"]:
            raise Exception("invalid dataset_name" % (dataset_name))
        self.name = dataset_name

    def data(self, image_size, input_nchan, batch_size=1):
        images = tf.ones(shape=[batch_size, image_size, image_size,
                         input_nchan], dtype=tf.float32, name="images")
        labels = tf.zeros(shape=[batch_size], dtype=tf.float32, name="labels")
        labels = tf.cast(labels, tf.int64, name="labels_cast")
        self.batch_size = batch_size
        return images, labels


def _get_model_map(dataset_name):
    if 'cifar10' == dataset_name:
        return _model_name_to_cifar_model
    elif dataset_name in ('imagenet', 'synthetic'):
        return _model_name_to_imagenet_model
    else:
        raise ValueError('Invalid dataset name: %s' % dataset_name)


def get_model_config(model_name, dataset):
    """Map model name to model network configuration."""
    model_map = _get_model_map(dataset.name)
    if model_name not in model_map:
        raise ValueError('Invalid model name \'%s\' for dataset \'%s\'' %
                         (model_name, dataset.name))
    else:
        return model_map[model_name]()


def build_convNet(CNN_model, dataset, batchsize=None):

    if dataset.name == 'cifar10':
        input_nchan = 3
        num_classes = 10
    else:
        input_nchan = 3
        num_classes = 1000

    model = get_model_config(CNN_model, dataset)
    # use specified batch_size if not None
    if batchsize is not None:
        model.set_batch_size(batchsize)

    images, labels = dataset.data(model.image_size, input_nchan,
                                  model.batch_size)
    network = convnet_builder.ConvNetBuilder(
        images, input_nchan, True, True,
        'NHWC', tf.float32, tf.float32)
    model.add_inference(network)
    logits = (network.affine(num_classes, activation='linear')
              if not model.skip_final_affine_layer()
              else network.top_layer)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits,
        name="cross_entropy_per_example")
    loss = tf.reduce_mean(cross_entropy, name="cross_entropy")
    opt = tf.train.GradientDescentOptimizer(0.123456)  # lr
    grads = opt.compute_gradients(loss)
    final_op = opt.apply_gradients(grads)
    return final_op, loss


def build_convNet_horovod(CNN_model, dataset, batchsize=None):

    if dataset.name == 'cifar10':
        input_nchan = 3
        num_classes = 10
    else:
        input_nchan = 3
        num_classes = 1000

    model = get_model_config(CNN_model, dataset)
    # use specified batch_size if not None
    if batchsize is not None:
        model.set_batch_size(batchsize)

    images, labels = dataset.data(model.image_size, input_nchan,
                                  model.batch_size)
    network = convnet_builder.ConvNetBuilder(
        images, input_nchan, True, True,
        'NHWC', tf.float32, tf.float32)
    model.add_inference(network)
    logits = (network.affine(num_classes, activation='linear')
              if not model.skip_final_affine_layer()
              else network.top_layer)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits,
        name="cross_entropy_per_example")
    loss = tf.reduce_mean(cross_entropy, name="cross_entropy")

    hvd.init()
    print("hvd.size(): %d" % hvd.size())

    opt = tf.train.GradientDescentOptimizer(0.123456 * hvd.size())  # lr
    opt = hvd.DistributedOptimizer(opt)

    grads = opt.compute_gradients(loss)
    final_op = opt.apply_gradients(grads)
    return final_op, loss
