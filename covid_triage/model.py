import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet50
from functools import partial

from dpipe import layers
from dpipe.torch import moveaxis
from dpipe.im.slices import iterate_axis
from dpipe.itertools import squeeze_first


def get_lungs_sgm_model():
    return nn.Sequential(
        nn.Conv2d(1, 8, kernel_size=3, padding=1),
        layers.FPN(
            layer=layers.ResBlock2d,
            downsample=nn.MaxPool2d(2, ceil_mode=True),
            upsample=nn.Identity,
            merge=lambda left, down: torch.add(*layers.interpolate_to_left(left, down, 'bilinear')),
            structure=[
                [[8, 8, 8], [8, 8, 8]],
                [[8, 16, 16], [16, 16, 8]],
                [[16, 32, 32], [32, 32, 16]],
                [[32, 64, 64], [64, 64, 32]],
                [[64, 128, 128], [128, 128, 64]],
                [[128, 256, 256], [256, 256, 128]],
                [256, 512, 256]
            ],
            kernel_size=3,
            padding=1
        ),
        layers.PreActivation2d(8, 1, kernel_size=1, bias=False)
    )


class MultitaskSpatial(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            layers.FPN(
                layer=layers.ResBlock2d,
                downsample=nn.MaxPool2d(2, ceil_mode=True),
                upsample=nn.Identity,
                merge=lambda left, down: torch.add(*layers.interpolate_to_left(left, down, 'bilinear')),
                structure=[
                    [[8, 8, 8], [8, 8]],
                    [[8, 16, 16], [16, 16, 8]],
                    [[16, 32, 32], [32, 32, 16]],
                    [[32, 64, 64], [64, 64, 32]],
                    [[64, 128, 128], [128, 128, 64]],
                    [[128, 256, 256], [256, 256, 128]],
                    [256, 512, 256]
                ],
                kernel_size=3,
                padding=1
            ),
        )
        sgm_head = nn.Sequential(
            layers.ResBlock2d(8, 8, kernel_size=3, padding=1),
            layers.PreActivation2d(8, 1, kernel_size=1, bias=False),  # shape (N, 1, H, W, D)
        )
        cls_head = nn.Sequential(
            layers.PyramidPooling(partial(F.max_pool3d, ceil_mode=True), levels=4),
            nn.Dropout(),
            nn.Linear(layers.PyramidPooling.get_multiplier(levels=4, ndim=3) * 8, 1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, 1, bias=False),  # shape (N, 1)
        )
        model = nn.Sequential(SliceWise(backbone), layers.Split(SliceWise(sgm_head), cls_head))

        self.backbone = backbone
        self.sgm_head = sgm_head
        self.cls_head = cls_head
        self.model = model
    
    def forward(self, x):
        return self.model(x)


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


def get_resnet():
    torchvision_resnet = resnet50()
    torchvision_resnet.conv1 = nn.Conv2d(1, torchvision_resnet.conv1.out_channels, 
        kernel_size=7, stride=2, padding=3, bias=False)

    return nn.Sequential(
        SliceWise(torchvision_resnet),  # (N, 1000, D)
        layers.Lambda(lambda x: x.contiguous()),
        layers.PyramidPooling(F.max_pool1d, levels=4),
        nn.Dropout(),
        nn.Linear(layers.PyramidPooling.get_multiplier(levels=4, ndim=1) * 1000, 1024),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(1024, 1),  # (N, 1)
    )
