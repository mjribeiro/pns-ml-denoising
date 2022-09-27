import numpy as np


class NoiseSource:
    def generate(self) -> np.ndarray:
        raise NotImplemented

    def apply(self, signal_length: np.ndarray) -> np.ndarray:
        raise NotImplemented