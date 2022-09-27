from .noise_source import NoiseSource

import numpy as np
from numpy.fft import irfft, rfftfreq


class BrownianNoise(NoiseSource):
    def __init__(self, dimensions: int, alpha: int = 2, f_min: float = 0):
        self.dimensions = dimensions
        self.signal_length = self.dimensions # copy of dimensions variable
        self.alpha = alpha
        self.f_min = f_min

    # generate() function adapted from: https://pypi.org/project/colorednoise/
    def generate(self) -> np.ndarray:

        # Make sure size is a list so we can iterate it and assign to it.
        try:
            self.dimensions = list(self.dimensions)
        except TypeError:
            self.dimensions = [self.dimensions]

        frequencies = rfftfreq(self.signal_length)

        # Build scaling factors for all frequencies
        s_scale = frequencies
        self.f_min = max(self.f_min, 1. / self.signal_length)  # Low frequency cutoff

        ix = np.sum(s_scale < self.f_min)  # Index of the cutoff
        if ix and ix < len(s_scale):
            s_scale[:ix] = s_scale[ix]
        s_scale = s_scale ** (-self.alpha / 2.)

        # Calculate theoretical output standard deviation from scaling
        w = s_scale[1:].copy()
        w[-1] *= (1 + (self.signal_length % 2)) / 2.  # correct f = +-0.5
        sigma = 2 * np.sqrt(np.sum(w ** 2)) / self.signal_length

        # Adjust dimensions to generate one Fourier component per frequency
        self.dimensions[-1] = len(frequencies)

        # Add empty dimension(s) to broadcast s_scale along last
        # dimension of generated random power + phase (below)
        dimensions_to_add = len(self.dimensions) - 1
        s_scale = s_scale[(np.newaxis,) * dimensions_to_add + (Ellipsis,)]

        # Generate scaled random power + phase
        sr = np.random.normal(scale=s_scale, size=self.dimensions)
        si = np.random.normal(scale=s_scale, size=self.dimensions)

        # If the signal length is even, frequencies +/- 0.5 are equal
        # so the coefficient must be real.
        if not (self.signal_length % 2): si[..., -1] = 0

        # Regardless of signal length, the DC component must be real
        si[..., 0] = 0

        # Combine power + corrected phase to Fourier components
        s = sr + 1J * si

        # Transform to real time series & scale to unit variance
        noise_signal = irfft(s, n=self.signal_length, axis=-1) / sigma
        return noise_signal

    def apply(self, input_signal: np.ndarray) -> np.ndarray:
        noise_signal = self.generate()
        return input_signal + noise_signal
