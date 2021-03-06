import abc

import numpy as np
import torch.utils.data


class RVDataset(torch.utils.data.Dataset, metaclass=abc.ABCMeta):
    """Random Variable Dataset abstract class."""
    def __init__(self, num_samples, shape=1, static_sample=False):
        assert isinstance(shape, (int, tuple))
        if isinstance(shape, int):
            shape = (1, shape)
        self.shape = shape
        self.num_samples = num_samples
        if static_sample:
            self.samples = [self._get_sample(i) for i in range(num_samples)]
        else:
            self.samples = None

    def __getitem__(self, item):
        if self.samples:
            return self.samples[item]
        return self._get_sample(item)

    def __len__(self):
        return self.num_samples

    @abc.abstractmethod
    def _get_sample(self, item):
        pass


class UniformRVDataset(RVDataset):
    """Uniform Random Variable Dataset."""
    def __init__(self, low=-1, high=1, **kwargs):
        super().__init__(**kwargs)
        self.low = low
        self.high = high

    def _get_sample(self, item):
        return np.random.uniform(self.low, self.high, size=self.shape)


class NormalRVDataset(RVDataset):
    """Normal Random Variable Dataset."""
    def __init__(self, mean=0, variance=1, **kwargs):
        self.mean = mean
        self.variance = variance
        super().__init__(**kwargs)

    def _get_sample(self, item):
        return np.random.normal(self.mean, self.variance, size=self.shape)


if __name__ == '__main__':
    uniform = UniformRVDataset(10, 2)
    normal = NormalRVDataset(10)
    for s in zip(iter(uniform), iter(normal)):
        print(s)
