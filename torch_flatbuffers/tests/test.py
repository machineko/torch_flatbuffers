import pytest
from torch_flatbuffers.export import Parser as parse_exp
from torch_flatbuffers.importer import Parser as parse_im
import torch
from copy import deepcopy
from torch import nn
import numpy as np
from torch_flatbuffers.schemes.Layers import *
from torch_flatbuffers.schemes.Layer import *


def test_all():
    parser = parse_exp(save_path="elo", name="conv2dsimple")
    module = nn.Sequential(
        nn.Conv2d(3, 6, (1, 1), bias=False),
        nn.Conv2d(6, 3, (2, 2), bias=True),
        nn.BatchNorm2d(3),
        nn.MaxPool2d(kernel_size=2, stride=2),
        # nn.Flatten(start_dim=2),
        nn.AdaptiveAvgPool2d((24, 24)),
        nn.Flatten(start_dim=2),
        nn.Conv1d(in_channels=3, out_channels=3, kernel_size=(1,)),
        nn.Flatten(),
        nn.Linear(in_features=24 * 24 * 3, out_features=3, bias=False),
    )
    inp = torch.rand(1, 3, 256, 256)
    out = module(inp)
    parser.parse_module(module=deepcopy(module), name="testconv")
    parser.save_to_flatbuff()

    buf = open("elo/conv2dsimple.data", "rb").read()
    buf = bytearray(buf)
    layers = Layers.GetRootAs(buf, 0)
    parser = parse_im()
    module = parser.parse_layers(layers)
    assert torch.allclose(module(inp), out)
