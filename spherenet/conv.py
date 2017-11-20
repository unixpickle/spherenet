"""
Spherical convolutional layers.
"""

import tensorflow as tf

def cos2d(inputs, filters, strides, padding):
    """
    Compute the 2-D convolutional cosine distances between
    the filters and the input patches.

    This mimics tf.nn.conv2d.

    Args:
      inputs: a 4-D input Tensor.
      filters: a 4-D filter Tensor.
      strides: a 1-D stride Tensor of length 4.
      padding: a string, 'SAME', or 'VALID'.

    Returns:
      A Tensor with filter-patch cosine distances along
      the inner-most dimension.
    """
    filter_height, filter_width, _, _ = filters.get_shape()
    patches = tf.extract_image_patches(images=inputs,
                                       ksizes=[1, filter_height, filter_width,
                                               inputs.get_shape()[-1]],
                                       strides=strides,
                                       rates=[1, 1, 1, 1],
                                       padding=padding)
    norm_patches = patches / tf.norm(patches, axis=-1, keep_dims=True)
    norm_filters = tf.reshape(filters, (-1, tf.shape(filters)[-1]))
    norm_filters /= tf.norm(norm_filters, axis=-1, keep_dims=True)
    return tf.matmul(norm_patches, norm_filters)
