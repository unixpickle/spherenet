"""
Tests for spherical convolutions.
"""

import unittest

import numpy as np
import tensorflow as tf

from .conv import cos2d

class Cos2DTest(unittest.TestCase):
    """
    Tests for the cos2d() primitive.
    """
    def test_single_patch(self):
        """
        Test when a single square patch fills up the
        entire input image.
        """
        image = np.random.normal(size=(3, 4, 4, 2))
        filters = np.random.normal(size=(4, 4, 2, 5))
        expected = np.zeros((3, 1, 1, 5), dtype='float64')
        for image_idx in range(image.shape[0]):
            for filter_idx in range(filters.shape[-1]):
                vec1 = image[image_idx, :, :, :].flatten()
                vec1 /= np.linalg.norm(vec1)
                vec2 = filters[:, :, :, filter_idx].flatten()
                vec2 /= np.linalg.norm(vec2)
                expected[image_idx, 0, 0, filter_idx] = np.dot(vec1, vec2)
        actual = cos2d(tf.constant(image), tf.constant(filters), [1, 1, 1, 1], 'VALID')
        with tf.Session() as sess:
            self.assertTrue(np.allclose(expected, sess.run(actual)))

if __name__ == '__main__':
    unittest.main()
