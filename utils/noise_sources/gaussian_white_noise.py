from .noise_source import NoiseSource

import numpy as np


class GaussianWhiteNoise(NoiseSource):

    def __init__(self, signal_length: int, std: float = 1, mean: float = 0):
        self.mean = mean
        self.std = std
        self.signal_length = signal_length

    def generate(self):
        return np.random.normal(self.mean, self.std, size=self.signal_length)

    def apply(self, input_signal: np.ndarray):
        noise_signal = self.generate()
        return input_signal + noise_signal
