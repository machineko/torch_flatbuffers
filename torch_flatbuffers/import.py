import time
from dataclasses import dataclass, field
from pathlib import Path
import torch
from torch import nn
import numpy as np
from torch_flatbuffers.schemes.Layers import *
from torch_flatbuffers.schemes.Layer import *
import flatbuffers



def parse_extras(params: dict, layer, keys: list[str]):
    for key in keys:
        if key == "weights":
            params["weights"] = layer.weight.numpy().ravel().astype(np.float32)
        elif key == "weightsShape":
            params["weightsShape"] = np.array(layer.weight.numpy().shape).astype(
                np.int32
            )
        elif key == "bias":
            params["bias"] = layer.bias.numpy().ravel().astype(np.float32)
        elif key == "biasShape":
            params["biasShape"] = np.array(layer.bias.numpy().shape).astype(np.int32)
        elif key == "numBatchesTracked":
            params["numBatchesTracked"] = (
                layer.num_batches_tracked.numpy().astype(np.float32).ravel()
            )
        elif key == "runningMean":
            params["runningMean"] = (
                layer.running_mean.numpy().astype(np.float32).ravel()
            )
        elif key == "runningMeanShape":
            params["runningMeanShape"] = np.array(
                layer.running_mean.numpy().shape
            ).astype(np.int32)
        elif key == "runningVar":
            params["runningVar"] = layer.running_var.numpy().astype(np.float32).ravel()
        elif key == "runningVarShape":
            params["runningVarShape"] = np.array(
                layer.running_var.numpy().shape
            ).astype(np.int32)
    return params


def load_conv2d(
    layer: Layer
) -> dict:

    conv = nn.Conv2d(
        in_channels=layer.InChannels(), out_channels=layer.OutChannels(), kernel_size=tuple(layer.KernelSizeAsNumpy()),
        stride=tuple(layer.StrideAsNumpy()), padding=tuple(layer.PaddingAsNumpy()), dilation=tuple(layer.DilationAsNumpy()),
        groups=layer.Groups(), bias=not layer.BiasIsNone()
    )
    conv.weight = nn.Parameter(torch.from_numpy(layer.WeightsAsNumpy()).reshape(tuple(layer.WeightsShapeAsNumpy())))
    if not layer.BiasIsNone():
        conv.bias = nn.Parameter(torch.from_numpy(layer.BiasAsNumpy()).reshape(tuple(layer.BiasShapeAsNumpy())))

    return conv


buf = open('elo/conv2dsimple.data', 'rb').read()
buf = bytearray(buf)
layers = Layers.GetRootAs(buf, 0)
module = load_conv2d(layer=layers.Layers(0))
inp = torch.load("inp.pt")
y = torch.load("out.pt")
print(y)
print(torch.allclose(module(inp), y))


# print(monster)
# print(monster.Layers(0).Type())
# for i in range(0, monster.LayersLength()):
#     layer = monster.Layers(i)
#     layer.InChannels
#     print(str(layer.Type()), layer.WeightsAsNumpy(), layer.WeightsAsNumpy().shape, layer.WeightsShapeAsNumpy())
#     load_conv2d(layer)