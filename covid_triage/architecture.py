import torch
from torch import nn

from dpipe.itertools import squeeze_first
from dpipe.torch import moveaxis


class SliceWise(nn.Module):
    def __init__(self, network: nn.Module, axis=-1):
        super().__init__()
        self.network = network
        self.axis = axis

    def forward(self, xs):
        bs, n_slices = len(xs), xs.shape[self.axis]
        # join self.axis with batch dim
        xs = moveaxis(xs, self.axis, 1)
        xs = xs.reshape(-1, *xs.shape[2:])

        xs = self.network(xs)
        # handling multiple outputs
        if isinstance(xs, torch.Tensor):
            xs = xs,

        # move self.axis back
        results = []
        for x in xs:
            x = x.reshape(bs, n_slices, *x.shape[1:])
            x = moveaxis(x, 1, self.axis)
            results.append(x)

        return squeeze_first(results)
