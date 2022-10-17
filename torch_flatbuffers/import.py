# type: ignore

import time
from dataclasses import dataclass, field
from pathlib import Path
import torch
from torch import nn
import numpy as np
from torch_flatbuffers.schemes.Layers import *
from torch_flatbuffers.schemes.Layer import *
import flatbuffers


def load_weights_bias(module, layer: Layer):
    module.weight = nn.Parameter(torch.from_numpy(layer.WeightsAsNumpy()).reshape(tuple(layer.WeightsShapeAsNumpy()))) 
    if not layer.BiasIsNone():
        module.bias = nn.Parameter(torch.from_numpy(layer.BiasAsNumpy()).flatten()) 

def load_running_vars(module, layer: Layer):
    
    module.running_mean = torch.from_numpy(layer.RunningMeanAsNumpy())
    module.running_var =  torch.from_numpy(layer.RunningVarAsNumpy())
    module.num_batches_tracked = torch.tensor(layer.NumBatchesTracked())
    module.num_features = layer.RunningVarAsNumpy().shape[0]

def load_kernel_stride(layer: Layer) -> tuple:
    return tuple(layer.KernelSizeAsNumpy()), tuple(layer.StrideAsNumpy())

def load_dilation(layer: Layer) -> tuple:
    return tuple(layer.DilationAsNumpy())

def load_padding(layer: Layer) -> tuple:
    return tuple(layer.PaddingAsNumpy())

def load_conv2d(
    layer: Layer
) -> nn.Conv2d:
    kernel_size, stride = load_kernel_stride(layer)

    conv = nn.Conv2d( 
        in_channels=layer.InChannels(), out_channels=layer.OutChannels(), kernel_size=kernel_size,
        stride=stride, padding=load_padding(layer), dilation=load_dilation(layer),
        groups=layer.Groups(), bias=not layer.BiasIsNone()
    )
    load_weights_bias(conv, layer)
    return conv

def load_batch_norm2d(layer: Layer) -> nn.BatchNorm2d:
    batch_norm = nn.BatchNorm2d(num_features=0, eps=layer.Eps(), momentum=layer.Momentum())
    load_weights_bias(batch_norm, layer)
    load_running_vars(batch_norm, layer)
    return batch_norm

def load_maxpool2d(
    layer: Layer
) -> nn.MaxPool2d:
    kernel_size, stride = load_kernel_stride(layer)
 
    max_pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, dilation=load_dilation(layer), 
                            padding=load_padding(layer))
    return max_pool

def load_flatten(layer: Layer) -> nn.Flatten:
    return nn.Flatten(start_dim=layer.StartDim(), end_dim=layer.EndDim())

def load_adaptive_avg_pool2d(layer: Layer) -> nn.AdaptiveAvgPool2d:
    return nn.AdaptiveAvgPool2d(output_size=tuple(layer.OutSizeAsNumpy()))

def load_linear(layer: Layer) -> nn.Linear:
    linear = nn.Linear(in_features=1, out_features=1, bias=False)
    load_weights_bias(linear, layer)
    linear.weight = nn.Parameter(linear.weight.T) 
    return linear

class Parser:

    @staticmethod
    def layer_switch(layer: Layer):
        match layer.Type().decode("utf-8"):
            case "BatchNorm2D":
                return load_batch_norm2d(layer)
            case "Conv2D":
                return load_conv2d(layer)
            case "MaxPool2D":
                return load_maxpool2d(layer)
            case "Flatten":
                return load_flatten(layer)
            case "AdaptiveAvgPool2D":
                return load_adaptive_avg_pool2d(layer)
            case "Linear":
                return load_linear(layer)
            case other:
                print(layer.Type().decode("utf-8"))
            # case "Dropout":
            #     return Dropout(fromTorchLayer: layer, graph: graph)



    def parse_layers(self, layers: Layers):
        layer_nb = layers.LayersLength()
        all_modules = []
        for i in range(layer_nb):
            all_modules.append(self.layer_switch(layers.Layers(i)))
        return nn.Sequential(*all_modules)

buf = open('elo/conv2dsimple.data', 'rb').read()
buf = bytearray(buf)
layers = Layers.GetRootAs(buf, 0)
parser = Parser()
module = parser.parse_layers(layers)
inp = torch.load("inp.pt")
y = torch.load("out.pt")
print(torch.allclose(module(inp), y))
print(y.mean(), module(inp).mean())
print(module(inp).shape, y.shape)

# print(monster)
# print(monster.Layers(0).Type())
# for i in range(0, monster.LayersLength()):
#     layer = monster.Layers(i)
#     layer.InChannels
#     print(str(layer.Type()), layer.WeightsAsNumpy(), layer.WeightsAsNumpy().shape, layer.WeightsShapeAsNumpy())
#     load_conv2d(layer)