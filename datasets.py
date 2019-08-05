import abc

import numpy as np
import torch.utils.data

np.random.seed(0)
torch.manual_seed(0)


class RVDataset(torch.utils.data.Dataset, metaclass=abc.ABCMeta):
    """Random Variable Dataset metaclass."""
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
    def _get_sample(self, item):
        return np.random.uniform(-0.5, 0.5, size=self.shape)


class NormalRVDataset(RVDataset):
    """Normal Random Variable Dataset."""
    def _get_sample(self, item):
        return np.random.normal(size=self.shape)


if __name__ == '__main__':
    uniform = UniformRVDataset(10)
    normal = NormalRVDataset(10)
    for s in zip(iter(uniform), iter(normal)):
        print(s)
