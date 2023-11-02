import unittest
import sys
import numpy as np
import numpy.random as rnd
from textural_image_features.features_extractor import FeaturesExtractor


class TestFeaturesExtractor(unittest.TestCase):
    def test_angular_second_moment(self):
        N_g = 256
        matrix = rnd.randint(0, N_g**2, size=(N_g, N_g))
        featuresExtractor = FeaturesExtractor.from_array(matrix)
        self.assertEqual(featuresExtractor.angular_second_moment, np.sum(matrix**2))

    def test_sum_entropy(self):
        N_g = 256
        matrix = rnd.randint(0, N_g**2, size=(N_g, N_g))
        featuresExtractor = FeaturesExtractor.from_array(matrix)

        def sum_when_sum_of_indices_equal(gtsdm, k):
            return np.fliplr(gtsdm).diagonal(gtsdm.shape[0] - 1 - k).sum()

        def sum_entropy_at(gtsdm, i):
            ith_sum = sum_when_sum_of_indices_equal(gtsdm, i)
            return ith_sum * np.log(ith_sum + sys.float_info.epsilon)

        gen_mapper = (
            sum_entropy_at(matrix, i) for i in np.arange(1, 2 * (N_g - 1) + 1)
        )
        self.assertAlmostEqual(
            featuresExtractor.sum_entropy,
            -np.sum(np.fromiter(gen_mapper, dtype=np.float64)),
        )

    def test_aliases(self):
        N_g = 256
        matrix = rnd.randint(0, N_g**2, size=(N_g, N_g))
        featuresExtractor = FeaturesExtractor.from_array(matrix)
        self.assertEqual(featuresExtractor.entropy, featuresExtractor.HXY)

    def test_exceptions(self):
        N_g = 256
        matrix = rnd.randint(0, N_g**2, size=(N_g, N_g))
        self.assertRaises(TypeError, FeaturesExtractor, matrix)


if __name__ == "__main__":
    unittest.main()
