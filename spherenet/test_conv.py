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
                vec1 = image[image_idx, :, :, :]
                vec2 = filters[:, :, :, filter_idx]
                expected[image_idx, 0, 0, filter_idx] = cos_dist(vec1, vec2)
        actual = cos2d(tf.constant(image), tf.constant(filters), [1, 1, 1, 1], 'VALID')
        with tf.Session() as sess:
            self.assertTrue(np.allclose(expected, sess.run(actual)))

    def test_asym_slided(self):
        """
        Test weirdly shaped patches with strides.
        """
        image = np.random.normal(size=(5, 4, 5, 2))
        filters = np.random.normal(size=(2, 3, 2, 7))
        expected = np.zeros((5, 2, 2, 7), dtype='float64')
        for image_idx in range(expected.shape[0]):
            for filter_idx in range(expected.shape[3]):
                img = image[image_idx, :, :, :]
                filt = filters[:, :, :, filter_idx]
                expected[image_idx, 0, 0, filter_idx] = cos_dist(filt, img[0:2, 0:3])
                expected[image_idx, 1, 0, filter_idx] = cos_dist(filt, img[2:4, 0:3])
                expected[image_idx, 0, 1, filter_idx] = cos_dist(filt, img[0:2, 2:5])
                expected[image_idx, 1, 1, filter_idx] = cos_dist(filt, img[2:4, 2:5])
        actual = cos2d(tf.constant(image), tf.constant(filters), [1, 2, 2, 1], 'VALID')
        with tf.Session() as sess:
            self.assertTrue(np.allclose(expected, sess.run(actual)))

def cos_dist(vec1, vec2):
    """
    Compute the cosine distance between two tensors.
    """
    vec1 = vec1.flatten()
    vec2 = vec2.flatten()
    vec1 = vec1 / np.linalg.norm(vec1)
    vec2 = vec2 / np.linalg.norm(vec2)
    return np.dot(vec1, vec2)

if __name__ == '__main__':
    unittest.main()
