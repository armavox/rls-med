from abc import ABC, abstractmethod
from typing import List, Union
import numpy as np
import torch


class CustomTransformNumpy(ABC):
    """Abstract method for custom numpy transformations.
    Every subclass should implement `__init__` for
    transformations parameters setting and `__call__` method for application to image.
    """

    @abstractmethod
    def __init__(self):
        raise NotImplementedError

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class Normalization(CustomTransformNumpy):
    def __init__(
        self,
        sample: List[np.ndarray] = None,
        from_min: float = None,
        from_max: float = None,
        to_min: float = None,
        to_max: float = None,
    ):
        self.to_min, self.to_max = to_min, to_max
        self.to_span = self.to_max - self.to_min

        if sample:
            sample = np.concatenate(sample)
            self.from_min = np.min(sample)
            self.from_max = np.max(sample)
        else:
            assert (from_min is not None) and (from_max is not None)
            self.from_min = from_min
            self.from_max = from_max
        self.from_span = self.from_max - self.from_min

    def __call__(self, volume: np.ndarray) -> np.ndarray:
        """ min max normalization"""
        scaled = (volume - self.from_min) / self.from_span
        return scaled * self.to_span + self.to_min

    def denorm(self, volume: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Denormalization with pre-saved stats"""
        scaled = (volume - self.to_min) / self.to_span
        return scaled * self.from_span + self.from_min
