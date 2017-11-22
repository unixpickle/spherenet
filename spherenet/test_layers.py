"""
Routines for easily creating spherical layers.
"""

# pylint: disable=E1129

import unittest

import numpy as np
import tensorflow as tf

from .layers import sphere_conv, ga_softmax
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
                                     regularization=0.1)
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
                sphere_conv(images, 4, 3, regularization=0.1)
                sphere_conv(images, 4, 3,
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            regularization=0.1)
                sess.run(tf.global_variables_initializer())
                orthog_reg, normal_reg = sess.run(tf.losses.get_regularization_losses())
                self.assertTrue(np.allclose(orthog_reg, 0))
                self.assertFalse(np.allclose(normal_reg, 0))

class GASoftmaxTest(unittest.TestCase):
    """
    Tests for the GA-Softmax.
    """
    def test_logits(self):
        """
        Test logit values in a very simple case.
        """
        for variant in ['linear', 'cosine', 'sigmoid']:
            with tf.Graph().as_default():
                with tf.Session() as sess:
                    inputs = tf.constant([[2]], dtype=tf.float64)
                    logits = ga_softmax(inputs, 2, variant=variant, margin=3)
                    sess.run(tf.global_variables_initializer())
                    weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                scope='ga_softmax/weights')[0]
                    sess.run(tf.assign(weights, np.array([[1, -1]], dtype='float64')))
                    actual = sess.run(logits)
                    expected = np.array([2, -2], dtype='float64')
                    self.assertTrue(np.allclose(actual, expected))

    def test_margin_loss(self):
        """
        Test the loss term with a margin.
        """
        scewl = tf.nn.softmax_cross_entropy_with_logits
        for variant in ['linear', 'cosine', 'sigmoid']:
            for margin in [1.5, 2, 2.5, 3, 3.5, 4]:
                with tf.Graph().as_default():
                    with tf.Session() as sess:
                        inputs = tf.constant([[1, 2], [3, 4]], dtype=tf.float64)
                        labels = tf.constant([[0.1, 0.7, 0.2], [0, 0, 1]], dtype=tf.float64)
                        logits, losses = ga_softmax(inputs, 3,
                                                    labels=labels,
                                                    variant=variant,
                                                    margin=margin)
                        sess.run(tf.global_variables_initializer())
                        weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                    scope='ga_softmax/weights')[0]
                        sess.run(tf.assign(weights, np.array([[0.5, 0.7, 0.3], [1, 1, -1]],
                                                             dtype='float64')))
                        logits, losses = sess.run((logits, losses))
                        logit_losses = sess.run(scewl(labels=labels, logits=logits))
                        self.assertTrue((logit_losses < losses).all())
