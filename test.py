import unittest
import numpy as np
from DmHaloGeometry.halo_orientation import HaloOrientation


class TestCalculations(unittest.TestCase):

    def test_get_eigs(self):
        test_I = np.array(([1, 0, 0], [0, 1, 0], [0, 0, 1]))
        test_length = 1

        e_val, e_vec = HaloOrientation.get_eigs(test_I, test_length)

        np.testing.assert_equal(e_vec, test_I, 'The eigenvectors are wrong.')
        np.testing.assert_equal(e_val, np.array((1, 1, 1)),
                                'The eigenvalues are wrong.')

    def test_fit_inertia_tensor(self):

        # create fake particle data
        coords = np.array((
            (1, 0, 0), (0, 1, 0), (0, 0, 1),
            (1, 0, 0), (0, 1, 0), (0, 0, 1),
            (1, 0, 0), (0, 1, 0), (0, 0, 1)
        ))

        x = coords.T[0]
        y = coords.T[1]
        z = coords.T[2]

        # calculate inertia tensor with max length 1
        # length used for calculating eigenvalues
        inertia_tensor = HaloOrientation.fit_inertia_tensor(
            np.array((x, y, z)), 1)

        test_I = np.array([[3., 0., 0.],
                           [0., 3., 0.],
                           [0., 0., 3.]])

        np.testing.assert_allclose(
            inertia_tensor, test_I, err_msg="The fit inertia tensors don't match")

    def test_get_perp_dist(self):

        # create fake particle data
        test_coords = np.array((
            (1, 0, 0), (0, 1, 0), (0, 0, 1),
        ))

        # inertia tensor corresponding to particles
        test_I = np.array([
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.]
        ])

        perp_component, angle = HaloOrientation.get_perp_dist(
            test_I, 1, test_coords)

        test_perp_component = np.array([[0., 0., 0.],
                                        [0., 1., 0.],
                                        [0., 0., 1.]])

        test_angle = np.array([0., 1.57079633, 1.57079633])

        np.testing.assert_allclose(
            test_perp_component, perp_component, err_msg="The fit inertia tensors don't match")
        np.testing.assert_allclose(
            test_angle, angle, err_msg="The fit inertia tensors don't match")


if __name__ == '__main__':
    unittest.main()
