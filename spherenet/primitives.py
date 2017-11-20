"""
Spherical convolutional layers.
"""

import tensorflow as tf

def cos1d(inputs, kernels, epsilon=1e-5):
    """
    Compute the 1-D cosine distances between a batch of
    inputs and a batch of kernels.

    Args:
      inputs: a 2-D Tensor of shape [batch x in_size].
      kernels: a 2-D Tensor of shape [in_size x out_size].
      epsilon: small number to prevent divisions by zero.

    Returns:
      A 2-D Tensor of shape [batch x out_size] containing
        cosine distances between each input vector and all
        the kernels.
    """
    norm_inputs = inputs / (tf.norm(inputs, axis=-1, keep_dims=True) + epsilon)
    norm_kernels = kernels / (tf.norm(kernels, axis=0) + epsilon)
    return tf.matmul(norm_inputs, norm_kernels)

def cos2d(inputs, filters, strides, padding, epsilon=1e-5):
    """
    Compute the 2-D convolutional cosine distances between
    the filters and the input patches.

    This mimics tf.nn.conv2d.

    Args:
      inputs: a 4-D input Tensor.
      filters: a 4-D filter Tensor.
      strides: a 1-D stride Tensor of length 4.
      padding: a string, 'SAME', or 'VALID'.
      epsilon: small number to prevent divisions by zero.

    Returns:
      A Tensor with filter-patch cosine distances along
      the inner-most dimension.
    """
    filter_height, filter_width, _, _ = filters.get_shape()
    patches = tf.extract_image_patches(images=inputs,
                                       ksizes=[1, filter_height, filter_width, 1],
                                       strides=strides,
                                       rates=[1, 1, 1, 1],
                                       padding=padding)
    norm_patches = patches / (tf.norm(patches, axis=-1, keep_dims=True) + epsilon)
    norm_filters = tf.reshape(filters, (-1, tf.shape(filters)[-1]))
    norm_filters /= tf.norm(norm_filters, axis=0, keep_dims=True) + epsilon
    return tf.einsum('abcd,de->abce', norm_patches, norm_filters)
