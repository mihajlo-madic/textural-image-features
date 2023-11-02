import unittest
import numpy as np
import numpy.random as rnd
from textural_image_features.gtsdmatrix import GTSDMatrix


class TestGTSDM(unittest.TestCase):
    def test_create(self):
        N_g = 256
        matrix = rnd.randint(0, N_g**2, size=(N_g, N_g))
        gtsdm = GTSDMatrix(matrix)
        self.assertIs(gtsdm.array, matrix)
        self.assertIsNone(gtsdm._P_X)
        self.assertIsNone(gtsdm._P_Y)
        self.assertIsNone(gtsdm._P_XY_SUM)
        self.assertIsNone(gtsdm._P_XY_DIFF)
        self.assertIsNone(gtsdm._HX)
        self.assertIsNone(gtsdm._HY)

    def test_exceptions(self):
        N_g = 2
        matrix = rnd.randint(0, N_g**2, size=(N_g, N_g))
        self.assertRaises(TypeError, GTSDMatrix, [1, 2, 3])
        self.assertRaises(ValueError, GTSDMatrix, matrix.ravel())


if __name__ == "__main__":
    unittest.main()
