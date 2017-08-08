import math
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import ops


class BatchNorm(object):
    """Code modification of http://stackoverflow.com/a/33950177"""

    def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum

            self.name = name

    def __call__(self, x, train):
        return tf.contrib.layers.batch_norm(x, decay=self.momentum,
                                            updates_collections=None,
                                            epsilon=self.epsilon,
                                            center=True, scale=True,
                                            is_training=train, scope=self.name)


def linear(inpt, output_dim, scope=None, stddev=1.0):
    """
    Linear transformation

    :param inpt: data
    :param output_dim: hidden layers
    :param scope: name
    :param stddev: standard deviation
    :return: tensor of type input
    """
    normal = tf.random_normal_initializer(stddev=stddev)
    const = tf.constant_initializer(0.0)
    with tf.variable_scope(scope or 'linear'):
        w = tf.get_variable('w', [inpt.get_shape()[1], output_dim],
                            initializer=normal)
        b = tf.get_variable('b', [output_dim], initializer=const)
        return tf.matmul(inpt, w) + b


def lrelu(x, leak=0.2, scope="lrelu"):
    """
    leaky relu

    if x > 0: return x
    else: return leak * x


    :param x: tensor
    :param leak: float, leak factor alpha >= 0
    :param scope: str, name of the operation
    :return: tensor, leaky relu operation
    """
    with tf.variable_scope(scope):
        # if leak < 1:
        #     return tf.maximum(x, leak * x)
        # elif x > 0:
        #     return x
        # else:
        #     return leak * x
        return tf.nn.relu(x) - leak * tf.nn.relu(-x)


def lrelu_alternative(x, leak=0.2, name="lrelu"):
    """
    Alternative implementation of lrelu

    :param x: tensor
    :param leak: float, leak factor alpha >= 0
    :param name: str, name of the operation
    :return: tensor, leaky relu operation
    """
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


def conv2d(inpt, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.2,
           scope=None):
    """
    Convolution for binned spike trains over the whole trial
    :param inpt:
    :param output_dim:
    :param k_h:
    :param k_w:
    :param d_h:
    :param d_w:
    :param stddev:
    :param scope:
    :return:
    """
    normal = tf.random_normal_initializer(stddev=stddev)
    const = tf.constant_initializer(0.0)
    with tf.variable_scope(scope or 'conv2d'):
        # TODO give an own defined filter
        w = tf.get_variable('w', [k_h, k_w, inpt.get_shape()[1], output_dim],
                            initializer=normal)
        conv = tf.nn.conv2d(inpt, w, strides=[1, d_h, d_w, 1], padding='SAME')
        b = tf.get_variable('b', [output_dim], initializer=const)
        # conv = tf.reshape(tf.nn.bias_add(conv, b), conv.get_shape())
        conv = tf.nn.bias_add(conv, b)
    return conv


def conv2d_transpose(inpt, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.2,
                     scope=None, with_w=False):
    initializer = tf.random_normal_initializer(stddev=stddev)
    const = tf.constant_initializer(0.0)
    with tf.variable_scope(scope or 'conv2d_transpose'):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_dim[1],
                                  inpt.get_shape()[1]],
                            initializer=initializer)
        deconv = tf.nn.conv2d_transpose(inpt, w, output_shape=output_dim,
                                        strides=[1, d_h, d_w, 1])
        biases = tf.get_variable('b', [output_dim[1]], initializer=const)
        # deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        deconv = tf.nn.bias_add(deconv, biases)
        if with_w:
            return deconv, w, biases
        else:
            return deconv


def conv1d(inpt, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.2,
           scope=None):
    """
    Convolution done for one binned spike train
    :param inpt:
    :param output_dim:
    :param k_h:
    :param k_w:
    :param d_h:
    :param d_w:
    :param stddev:
    :param scope:
    :return:
    """
    normal = tf.random_normal_initializer(stddev=stddev)
    const = tf.constant_initializer(0.0)
    with tf.variable_scope(scope or 'conv1d'):
        # TODO give an own defined filter
        w = tf.get_variable('w', [k_h, k_w, inpt.get_shape()[1], output_dim],
                            initializer=normal)
        conv = tf.nn.conv1d(inpt, w, stride=[1, d_h, d_w, 1], padding='SAME')
        b = tf.get_variable('b', [output_dim], initializer=const)
        # conv = tf.reshape(tf.nn.bias_add(conv, b), conv.get_shape())
        conv = tf.nn.bias_add(conv, b)
    return conv


def binary_cross_entropy(preds, targets, scope=None):
    """Computes binary cross entropy given `preds`.
    For brevity, let `x = `, `z = targets`.  The logistic loss is
        loss(x, z) = - sum_i (x[i] * log(z[i]) + (1 - x[i]) * log(1 - z[i]))
    Args:
        preds: A `Tensor` of type `float32` or `float64`.
        targets: A `Tensor` of the same type and shape as `preds`.
    """
    eps = 1e-12
    with ops.op_scope([preds, targets], scope, "bce_loss") as scope:
        preds = ops.convert_to_tensor(preds, name="preds")
        targets = ops.convert_to_tensor(targets, name="targets")
        return tf.reduce_mean(-(targets * tf.log(preds + eps) +
                                (1. - targets) * tf.log(1. - preds + eps)))


def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return tf.concat(3, [x, y * tf.ones(
        [x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])])
