"""
Routines for easily creating spherical layers.
"""

import math

import tensorflow as tf

from .primitives import cos2d

# pylint: disable=R0913
def sphere_conv(inputs,
                filters,
                kernel_size,
                strides=(1, 1),
                padding='valid',
                variant='linear',
                sigmoid_k=None,
                kernel_initializer=tf.orthogonal_initializer(),
                regularize=False,
                name='sphere_conv'):
    """
    Create a SphereConv layer.

    This is similar to tf.layers.conv2d.

    Args:
      inputs: a 4-D input Tensor. Order is NHWC.
      filters: number of output filters.
      kernel_size: kernel dimensions (int or sequence).
      strides: convolutional stride (int or sequence).
      padding: 'valid' or 'same'.
      variant: 'linear', 'cosine', or 'sigmoid'.
      sigmoid_k: the `k` parameter for sigmoid layers.
        If None, a trainable variable is used.
      kernel_initializer: initializer for the kernels.
      regularize: if True, a regularization term is added
        to tf.GraphKeys.REGULARIZATION_LOSSES.
      name: name of the layer.
    """
    if not tf.contrib.framework.nest.is_sequence(kernel_size):
        kernel_size = (kernel_size, kernel_size)
    if not tf.contrib.framework.nest.is_sequence(strides):
        strides = (strides, strides)
    with tf.variable_scope(None, default_name=name):
        in_depth = int(inputs.get_shape()[-1])
        kernels = tf.get_variable('kernels',
                                  dtype=inputs.dtype,
                                  shape=(kernel_size[0], kernel_size[1], in_depth, filters),
                                  initializer=kernel_initializer)
        if regularize:
            _add_kernel_regularizer(tf.reshape(kernels, (-1, filters)))
        cosines = cos2d(inputs, kernels, [1, strides[0], strides[1], 1], padding.upper())
        if variant == 'cosine':
            return cosines
        elif variant == 'linear':
            return 1 - (2/math.pi)*tf.acos(cosines)
        elif variant == 'sigmoid':
            sigmoid_k = (sigmoid_k or tf.get_variable('k',
                                                      dtype=kernels.dtype,
                                                      initializer=[0.5]*filters))
            return _sigmoid_nonlinearity(cosines, sigmoid_k)
        else:
            raise ValueError('unknown variant: ' + variant)

def _add_kernel_regularizer(matrix):
    """
    Creates a regularization loss that encourages the
    columns of the matrix to be orthogonal.
    """
    dots = tf.matmul(tf.transpose(matrix), matrix)
    ident = tf.eye(int(matrix.get_shape()[-1]), dtype=matrix.dtype)
    diffs = tf.reduce_sum(tf.square(dots - ident))
    tf.losses.add_loss(diffs, loss_collection=tf.GraphKeys.REGULARIZATION_LOSSES)

def _sigmoid_nonlinearity(cosines, sigmoid_k):
    """
    Compute a sigmoid SphereConv from angle cosines.
    """
    pi_coeff = -math.pi / (2 * sigmoid_k)
    scale = (1 + tf.exp(pi_coeff)) / (1 - tf.exp(pi_coeff))
    main_exp = tf.exp(tf.acos(cosines)/sigmoid_k + pi_coeff)
    return scale * (1 - main_exp) / (1 + main_exp)
