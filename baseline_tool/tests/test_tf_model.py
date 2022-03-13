from tf_profile.models import tf_model
import tensorflow as tf
import pytest


def test_seq2seq():
    # test for seq2seq
    with tf.Graph().as_default():
        assert tf_model.exist_model('seq2seq')
        op, loss = tf_model.get_model('seq2seq')

    with tf.Graph().as_default():
        assert tf_model.exist_model('seq2seq', horovod=True)
        op, loss = tf_model.get_model('seq2seq', horovod=True)


def test_nasnet():
    # test for nasnet
    with tf.Graph().as_default():
        assert tf_model.exist_model('nasnet')
        op, loss = tf_model.get_model('nasnet')

    with tf.Graph().as_default():
        assert tf_model.exist_model('nasnet', horovod=True)
        op, loss = tf_model.get_model('nasnet', horovod=True)


def test_deepspeech():
    # test for seq2seq
    with tf.Graph().as_default():
        assert tf_model.exist_model('deepspeech')
        op, loss = tf_model.get_model('deepspeech')

    with tf.Graph().as_default():
        assert tf_model.exist_model('deepspeech', horovod=True)
        op, loss = tf_model.get_model('deepspeech', horovod=True)


def test_lstm():
    # test for lstm
    with tf.Graph().as_default():
        assert tf_model.exist_model('lstm')
        op, loss = tf_model.get_model('lstm')

    with tf.Graph().as_default():
        assert tf_model.exist_model('lstm', horovod=True)
        op, loss = tf_model.get_model('lstm', horovod=True)


def test_resnet50():
    # test for resnet50
    with tf.Graph().as_default():
        assert tf_model.exist_model('resnet50')
        op, loss = tf_model.get_model('resnet50')

    with tf.Graph().as_default():
        assert tf_model.exist_model('resnet50', horovod=True)
        op, loss = tf_model.get_model('resnet50', horovod=True)


def test_vgg16():
    # test for vgg16
    with tf.Graph().as_default():
        assert tf_model.exist_model('vgg16')
        op, loss = tf_model.get_model('vgg16')

    with tf.Graph().as_default():
        assert tf_model.exist_model('vgg16', horovod=True)
        op, loss = tf_model.get_model('vgg16', horovod=True)


def test_alexnet():
    # test for vgg16
    with tf.Graph().as_default():
        assert tf_model.exist_model('alexnet')
        op, loss = tf_model.get_model('alexnet')

    with tf.Graph().as_default():
        assert tf_model.exist_model('alexnet', horovod=True)
        op, loss = tf_model.get_model('alexnet', horovod=True)


def test_inception3():
    # test for vgg16
    with tf.Graph().as_default():
        assert tf_model.exist_model('inception3')
        op, loss = tf_model.get_model('inception3')

    with tf.Graph().as_default():
        assert tf_model.exist_model('inception3', horovod=True)
        op, loss = tf_model.get_model('inception3', horovod=True)


def test_tf_model():
    # test for Error raising
    assert not tf_model.exist_model('foobar')
    with pytest.raises(ValueError):
        tf_model.get_model('foobar')
