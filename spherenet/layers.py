"""
Routines for easily creating spherical layers.
"""

from math import pi

import tensorflow as tf

from .primitives import cos1d, cos2d, sigmoid_nonlinearity

# pylint: disable=R0913
def sphere_conv(inputs,
                filters,
                kernel_size,
                strides=(1, 1),
                padding='valid',
                variant='linear',
                sigmoid_k=None,
                kernel_initializer=tf.orthogonal_initializer(),
                regularization=None,
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
        This value is broadcast as needed.
      kernel_initializer: initializer for the kernels.
      regularization: the regularization coefficient.
        If not none, a regularization term is added to
        tf.GraphKeys.REGULARIZATION_LOSSES.
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
        if regularization:
            _add_kernel_regularizer(tf.reshape(kernels, (-1, filters)), regularization)
        cosines = cos2d(inputs, kernels, [1, strides[0], strides[1], 1], padding.upper())
        if variant == 'cosine':
            return cosines
        elif variant == 'linear':
            return 1 - (2/pi)*tf.acos(cosines)
        elif variant == 'sigmoid':
            sigmoid_k = _sigmoid_k_or_default(sigmoid_k, inputs.dtype)
            return sigmoid_nonlinearity(tf.acos(cosines), sigmoid_k)
        else:
            raise ValueError('unknown variant: ' + variant)

# pylint: disable=R0914
def ga_softmax(inputs,
               outputs,
               labels=None,
               variant='linear',
               sigmoid_k=None,
               margin=1,
               initializer=tf.orthogonal_initializer(),
               name='ga_softmax'):
    """
    Create a generalized angular softmax layer.

    If `labels` is specified, a loss is computed and added
    to tf.GraphKeys.LOSSES.

    Args:
      inputs: a 2-D Tensor representing a batch of feature
        vectors to use for classification.
      outputs: number of output labels.
      labels: output labels for computing the loss.
        If specified, this is a batch of probability
        distributions.
      variant: 'linear', 'cosine', or 'sigmoid'.
      sigmoid_k: the `k` parameter for sigmoid layers.
        If None, a trainable variable is used.
        This value is broadcast as needed.
      margin: the margin coefficient. Values higher than 1
        encourage a large margin.
      initializer: initializer for the weight matrix.
      name: name of the layer.

    Returns:
      If no labels are specified, a Tensor of logits.
      If labels are specified, a pair (logits, losses).
      The resulting logits do not depend on the margin.
    """
    with tf.variable_scope(None, default_name=name):
        weights = tf.get_variable('weights',
                                  dtype=inputs.dtype,
                                  shape=(inputs.get_shape()[-1], outputs),
                                  initializer=initializer)
        angles = tf.acos(cos1d(inputs, weights))
        activation_fn = _ga_softmax_activation(variant, sigmoid_k, inputs.dtype)
        norms = tf.norm(inputs, axis=-1, keep_dims=True)
        logits = activation_fn(angles) * norms
        if labels is None:
            return logits
        margin_logits = activation_fn(angles * margin) * norms
        loss = None
        for i in range(outputs):
            sub_logits = tf.concat([logits[:, :i], margin_logits[:, i:i+1], logits[:, i+1:]],
                                   axis=-1)
            term = labels[:, i] * tf.nn.log_softmax(sub_logits)[:, i]
            if loss is None:
                loss = -term
            else:
                loss -= term
        tf.losses.add_loss(tf.reduce_mean(loss))
        return logits, loss

def _ga_softmax_activation(variant, sigmoid_k, dtype):
    """
    Get a function that applies the monotonically
    decreasing activation for a GA-Softmax.
    """
    if variant == 'linear':
        return lambda x: 1 - (2/pi)*x
    elif variant == 'cosine':
        return _repeated_cosine
    elif variant == 'sigmoid':
        sigmoid_k = _sigmoid_k_or_default(sigmoid_k, dtype)
        return lambda x: sigmoid_nonlinearity(x, sigmoid_k)
    else:
        raise ValueError('unknown variant: ' + variant)

def _add_kernel_regularizer(matrix, coeff):
    """
    Creates a regularization loss that encourages the
    columns of the matrix to be orthogonal.
    """
    dots = tf.matmul(tf.transpose(matrix), matrix)
    ident = tf.eye(int(matrix.get_shape()[-1]), dtype=matrix.dtype)
    diffs = tf.reduce_sum(tf.square(dots - ident))
    tf.losses.add_loss(diffs*coeff, loss_collection=tf.GraphKeys.REGULARIZATION_LOSSES)

def _sigmoid_k_or_default(sigmoid_k, dtype):
    """
    Return the sigmoid k constant or create a variable if
    the constant is None.
    """
    if sigmoid_k is not None:
        return sigmoid_k
    return tf.get_variable('k', dtype=dtype, initializer=tf.constant(0.5, dtype=dtype))

def _repeated_cosine(theta):
    """
    Monotonically decreasing piecewise cosine.
    """
    phase = tf.mod(theta, pi)
    offset = tf.floor(theta / pi)
    return tf.cos(phase) - 2*offset
