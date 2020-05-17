import unittest
import numpy as np
from periodicity.spectral import lombscargle, wavelet


class LombScargleTest(unittest.TestCase):
    def test_frequency_grid(self):
        t0 = 2.5
        ts = 0.1
        fs = 1 / ts
        f0 = 1 / t0
        time = np.arange(0, t0+ts, ts)
        signal = np.random.normal(1, 0.1, time.size)
        ls, freq, power = lombscargle(time, signal, n=1)
        self.assertListEqual(sorted(freq), list(freq))
        self.assertEqual(f0 / 2, freq[0])
        self.assertAlmostEqual(fs / 2, freq[-1])
        self.assertLess(np.max(np.abs(np.diff(freq) - f0)), 1e-10)

    def test_can_find_periods(self):
        time = np.arange(100)
        sine = np.sin((time / np.max(time)) * 20 * np.pi)
        ls, freq, power = lombscargle(time, sine)
        best_period = 1 / freq[power.argmax()]
        self.assertAlmostEqual(best_period, 10.0)


class WaveletTest(unittest.TestCase):
    def test_return_value_shape(self):
        time = np.arange(10)
        signal = np.random.normal(1, 0.1, time.size)
        periods = np.array([0.5, 1, 2])
        power, coi, mask_coi = wavelet(time, signal, periods)
        self.assertEqual(power.shape, (periods.size, time.size))
        self.assertEqual(coi[0].shape, coi[1].shape)
        self.assertEqual(mask_coi.shape, power.shape)


if __name__ == '__main__':
    unittest.main()
