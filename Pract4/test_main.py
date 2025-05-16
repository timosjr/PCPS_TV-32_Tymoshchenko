import unittest
import numpy as np

from main import simulate_lorenz


class Test(unittest.TestCase):
    def test_simulate_shape(self):
        xs, ys, zs = simulate_lorenz((0, 1, 1.05), dt=0.01, num_steps=100)
        self.assertEqual(len(xs), 101)
        self.assertEqual(len(ys), 101)
        self.assertEqual(len(zs), 101)

    def test_close_initial_conditions(self):
        xs1, ys1, zs1 = simulate_lorenz((0, 1, 1.05), dt=0.01, num_steps=100)
        xs2, ys2, zs2 = simulate_lorenz((0.00001, 1, 1.05), dt=0.01, num_steps=100)

        diff = np.sqrt((xs1 - xs2) ** 2 + (ys1 - ys2) ** 2 + (zs1 - zs2) ** 2)

        #спочатку різниця мала
        self.assertLess(diff[0], 1e-4)
        #потім має збільшитись (хоча час короткий, перевіримо, чи не стало 0)
        self.assertGreater(diff[-1], diff[0])

    def test_identical_initial_conditions(self):
        xs1, ys1, zs1 = simulate_lorenz((0, 1, 1.05), dt=0.01, num_steps=100)
        xs2, ys2, zs2 = simulate_lorenz((0, 1, 1.05), dt=0.01, num_steps=100)

        np.testing.assert_allclose(xs1, xs2, atol=1e-12)
        np.testing.assert_allclose(ys1, ys2, atol=1e-12)
        np.testing.assert_allclose(zs1, zs2, atol=1e-12)

if __name__ == '__main__':
    unittest.main()