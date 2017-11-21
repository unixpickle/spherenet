"""
Routines for easily creating spherical layers.
"""

# pylint: disable=E1129

import unittest

import numpy as np
import tensorflow as tf

from .layers import sphere_conv
from .primitives import cos2d

class SphereConvTest(unittest.TestCase):
    """
    Tests for sphere_conv layers.
    """
    def test_cosines(self):
        """
        Test the cosine distances produced by the layer.
        """
        with tf.Graph().as_default():
            with tf.Session() as sess:
                images = tf.random_normal((3, 8, 9, 5), dtype=tf.float64)
                actual = sphere_conv(images,
                                     filters=4,
                                     kernel_size=(3, 2),
                                     strides=2,
                                     padding='same',
                                     variant='cosine',
                                     regularize=True)
                self.assertEqual(actual.dtype, tf.float64)
                kernels = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                            scope="sphere_conv/kernels")[0]
                expected = cos2d(images, kernels, [1, 2, 2, 1], 'SAME')
                sess.run(tf.global_variables_initializer())
                actual, expected = sess.run((actual, expected))
                self.assertTrue(np.allclose(expected, actual))

    def test_regularizer(self):
        """
        Basic regularizer tests.
        """
        with tf.Graph().as_default():
            with tf.Session() as sess:
                images = tf.random_normal((3, 8, 9, 5), dtype=tf.float64)
                sphere_conv(images, 4, 3, regularize=True)
                sphere_conv(images, 4, 3,
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            regularize=True)
                sess.run(tf.global_variables_initializer())
                orthog_reg, normal_reg = sess.run(tf.losses.get_regularization_losses())
                self.assertTrue(np.allclose(orthog_reg, 0))
                self.assertFalse(np.allclose(normal_reg, 0))
