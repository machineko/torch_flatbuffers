# type: ignore // pylance not working with flatbuffers and some of the torch + numpy functions and objects

import time
from dataclasses import dataclass, field
from pathlib import Path
import torch
from torch import nn
import numpy as np
from torch_flatbuffers.schemes.Layers import *
from torch_flatbuffers.schemes.Layer import *
import flatbuffers

nn.Conv2d


def check_padding(layer):
    supported_pad = ["zeros", "reflect"]
    if layer.padding_mode not in supported_pad:
        raise ValueError(
            f"{layer.padding_mode} not supported only {supported_pad} supported"
        )


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


def build_basic_info(builder, idx: int, layer_type: str, name: str):
    LayerAddIdx(builder, idx)
    LayerAddType(builder, layer_type)
    LayerAddName(builder, name)


def build_weights_bias(
    builder, weights, weights_shape, extra_keys: dict, bias=None, bias_shape=None
):
    LayerAddWeights(builder, weights)
    LayerAddWeightsShape(builder, weights_shape)
    if "bias" in extra_keys:
        LayerAddBias(builder, bias)
        LayerAddBiasShape(builder, bias_shape)


def build_conv(
    builder, data_layout, dilation, kernel_size, padding, stride, pad_mode, params: dict
):
    LayerAddDataLayout(builder, data_layout)
    LayerAddDilation(builder, dilation)
    LayerAddKernelSize(builder, kernel_size)
    LayerAddPadding(builder, padding)
    LayerAddStride(builder, stride)
    LayerAddPadMode(builder, pad_mode)
    LayerAddInChannels(builder, params["in_channels"])
    LayerAddOutChannels(builder, params["out_channels"])
    LayerAddGroups(builder, params["groups"])


def create_weights_bias(builder, params: dict, extra_keys: dict):
    weights = builder.CreateNumpyVector(params["weights"])
    weights_shape = builder.CreateNumpyVector(params["weightsShape"])
    if "bias" in extra_keys:
        bias = builder.CreateNumpyVector(params["bias"])
        bias_shape = builder.CreateNumpyVector(params["biasShape"])
    else:
        bias, bias_shape = None, None
    return weights, weights_shape, bias, bias_shape


def save_conv2d(
    layer: nn.Conv2d, builder: flatbuffers.Builder, name: str, idx: int
) -> dict:
    keys = [
        "dilation2d",
        "kernel_size2d",
        "padding2d",
        "stride2d",
        "in_channels",
        "groups",
        "out_channels",
    ]
    params = {}
    layer_dict = layer.__dict__
    for k in keys:
        if k == "padding2d":
            if isinstance(layer.padding, str):
                assert layer.padding == "valid", "same not supported, yet :)"
                params[f"{k}"] = [0, 0]
                continue
            padding = layer.padding
            if len(padding) == 1:
                params[f"{k}"] = list(layer.padding) + list(layer.padding)
                continue
            elif len(padding) == 2:
                params[f"{k}"] = list(layer.padding)
                continue
            else:
                raise ValueError(f"Wrong shape {layer.padding}")

        tmp_k = k.replace("2d", "")
        if "2d" not in k:
            params[f"{k}"] = layer_dict[k]
        else:
            if not isinstance(layer_dict[tmp_k], (list, tuple)):
                params[f"{k}"] = [layer_dict[tmp_k], layer_dict[tmp_k]]
            else:
                params[f"{k}"] = layer_dict[tmp_k]
    extra_keys = ["weights", "weightsShape"]
    if isinstance(layer.bias, torch.Tensor):
        layer.bias = nn.Parameter(layer.bias[None, :, None, None])  # 4d
        extra_keys.append("bias")
        extra_keys.append("biasShape")

    check_padding(layer)

    params = parse_extras(params, layer, extra_keys)
    name = builder.CreateString(name)
    layer_type = builder.CreateString("Conv2D")
    data_layout = builder.CreateString("NCHW")
    pad_mode = builder.CreateString(layer.padding_mode)
    dilation = builder.CreateNumpyVector(
        np.asarray(params["dilation2d"]).astype(np.int32)
    )
    kernel_size = builder.CreateNumpyVector(
        np.asarray(params["kernel_size2d"]).astype(np.int32)
    )
    padding = builder.CreateNumpyVector(
        np.asarray(params["padding2d"]).astype(np.int32)
    )
    stride = builder.CreateNumpyVector(np.asarray(params["stride2d"]).astype(np.int32))

    weights, weights_shape, bias, bias_shape = create_weights_bias(
        builder, params, extra_keys
    )
    LayerStart(builder)

    build_basic_info(builder, idx, layer_type, name)
    build_conv(
        builder, data_layout, dilation, kernel_size, padding, stride, pad_mode, params
    )

    build_weights_bias(
        builder=builder,
        weights=weights,
        weights_shape=weights_shape,
        extra_keys=extra_keys,
        bias=bias,
        bias_shape=bias_shape,
    )

    return LayerEnd(builder)


def save_batchnorm2d(
    layer: nn.BatchNorm2d, builder: flatbuffers.Builder, name: str, idx: int
):
    keys = ["eps", "momentum"]
    params = {}
    layer_dict = layer.__dict__
    for k in keys:
        params[f"{k}"] = layer_dict[k]
    extra_keys = [
        "weights",
        "weightsShape",
        "bias",
        "biasShape",
        "numBatchesTracked",
        "runningMean",
        "runningMeanShape",
        "runningVar",
        "runningVarShape",
    ]
    layer.weight = nn.Parameter(layer.weight[None, :, None, None])
    layer.bias = nn.Parameter(layer.bias[None, :, None, None])

    layer.running_mean = nn.Parameter(layer.running_mean[None, :, None, None])
    layer.running_var = nn.Parameter(layer.running_var[None, :, None, None])

    params = parse_extras(params, layer, extra_keys)

    name = builder.CreateString(name)
    layer_type = builder.CreateString("BatchNorm2D")

    running_mean = builder.CreateNumpyVector(np.asarray(params["runningMean"]))
    running_mean_shape = builder.CreateNumpyVector(
        np.asarray(params["runningMeanShape"]).astype(np.int32)
    )

    running_var = builder.CreateNumpyVector(np.asarray(params["runningVar"]))
    running_var_shape = builder.CreateNumpyVector(
        np.asarray(params["runningVarShape"]).astype(np.int32)
    )

    weights, weights_shape, bias, bias_shape = create_weights_bias(
        builder, params, extra_keys
    )
    batch_tracked = int(params["numBatchesTracked"])
    eps = float(layer.eps)

    LayerStart(builder)

    build_basic_info(builder, idx, layer_type, name)

    LayerAddRunningMean(builder, running_mean)
    LayerAddRunningMeanShape(builder, running_mean_shape)

    LayerAddRunningVar(builder, running_var)
    LayerAddRunningVarShape(builder, running_var_shape)

    build_weights_bias(
        builder=builder,
        weights=weights,
        weights_shape=weights_shape,
        extra_keys=extra_keys,
        bias=bias,
        bias_shape=bias_shape,
    )

    LayerAddNumBatchesTracked(builder, batch_tracked)
    LayerAddEps(builder, eps)

    return LayerEnd(builder)


def save_maxpool2d(
    layer: nn.MaxPool2d, builder: flatbuffers.Builder, name: str, idx: int
) -> dict:
    keys = ["dilation2d", "kernel_size2d", "padding2d", "stride2d"]
    params = {}
    layer_dict = layer.__dict__
    for k in keys:
        tmp_k = k.replace("2d", "")
        if "2d" not in k:
            params[f"{k}"] = layer_dict[k]
        else:
            if not isinstance(layer_dict[tmp_k], (list, tuple)):
                params[f"{k}"] = [layer_dict[tmp_k], layer_dict[tmp_k]]
            else:
                params[f"{k}"] = layer_dict[tmp_k]

    name = builder.CreateString(name)
    layer_type = builder.CreateString("MaxPool2D")
    data_layout = builder.CreateString("NCHW")
    dilation = builder.CreateNumpyVector(
        np.asarray(params["dilation2d"]).astype(np.int32)
    )
    kernel_size = builder.CreateNumpyVector(
        np.asarray(params["kernel_size2d"]).astype(np.int32)
    )
    padding = builder.CreateNumpyVector(
        np.asarray(params["padding2d"]).astype(np.int32)
    )
    stride = builder.CreateNumpyVector(np.asarray(params["stride2d"]).astype(np.int32))

    LayerStart(builder)

    build_basic_info(builder, idx, layer_type, name)

    LayerAddDataLayout(builder, data_layout)
    LayerAddDilation(builder, dilation)
    LayerAddKernelSize(builder, kernel_size)
    LayerAddPadding(builder, padding)
    LayerAddStride(builder, stride)

    return LayerEnd(builder)


def save_adaptiveavgpool2d(
    layer: nn.AdaptiveAvgPool2d, builder: flatbuffers.Builder, name: str, idx: int
) -> dict:
    params = {}
    params["outSize2d"] = []
    if isinstance(layer.output_size, tuple):
        for i in layer.output_size:
            if i:
                params["outSize2d"].append(i)
            else:
                params["outSize2d"].append(-1)
    else:
        if isinstance(layer.output_size, int) or len(layer.output_size) == 1:
            params["outSize2d"] = [layer.output_size, layer.output_size]
        else:
            for i in layer.output_size:
                if i:
                    params["outSize2d"].append(i)
                else:
                    params["outSize2d"].append(-1)

    name = builder.CreateString(name)
    layer_type = builder.CreateString("AdaptiveAvgPool2D")
    data_layout = builder.CreateString("NCHW")
    out_size2d = builder.CreateNumpyVector(
        np.asarray(params["outSize2d"]).astype(np.int32)
    )
    LayerStart(builder)

    build_basic_info(builder, idx, layer_type, name)

    LayerAddDataLayout(builder, data_layout)
    LayerAddOutSize(builder, out_size2d)

    return LayerEnd(builder)


def save_relu(
    layer: nn.ReLU, builder: flatbuffers.Builder, name: str, idx: int
) -> dict:
    name = builder.CreateString(name)
    layer_type = builder.CreateString("ReLU")
    LayerStart(builder)

    build_basic_info(builder, idx, layer_type, name)

    return LayerEnd(builder)


def save_dropout(
    layer: nn.Dropout, builder: flatbuffers.Builder, name: str, idx: int
) -> dict:
    name = builder.CreateString(name)
    layer_type = builder.CreateString("Dropout")
    LayerStart(builder)

    build_basic_info(builder, idx, layer_type, name)

    LayerAddProbability(builder, layer.p)
    return LayerEnd(builder)


def save_flatten(layer: nn.Flatten, builder: flatbuffers.Builder, name: str, idx: int):
    name = builder.CreateString(name)
    layer_type = builder.CreateString("Flatten")
    LayerStart(builder)

    build_basic_info(builder, idx, layer_type, name)

    LayerAddStartDim(builder, layer.start_dim)
    LayerAddEndDim(builder, layer.end_dim)
    return LayerEnd(builder)


def save_linear(layer: nn.Linear, builder: flatbuffers.Builder, name: str, idx: int):
    params = {}
    extra_keys = ["weights", "weightsShape"]
    if isinstance(layer.bias, torch.Tensor):
        extra_keys.append("bias")
        extra_keys.append("biasShape")

    layer.weight = nn.Parameter(
        torch.transpose(layer.weight, 0, 1)
    )  # matmul without transpose
    params = parse_extras(params, layer, extra_keys)

    name = builder.CreateString(name)
    layer_type = builder.CreateString("Linear")

    weights = builder.CreateNumpyVector(params["weights"])
    weights_shape = builder.CreateNumpyVector(params["weightsShape"])

    weights, weights_shape, bias, bias_shape = create_weights_bias(
        builder, params, extra_keys
    )

    LayerStart(builder)

    build_basic_info(builder, idx, layer_type, name)

    build_weights_bias(
        builder=builder,
        weights=weights,
        weights_shape=weights_shape,
        extra_keys=extra_keys,
        bias=bias,
        bias_shape=bias_shape,
    )
    return LayerEnd(builder)


@dataclass
class Parser:
    save_path: str
    name: str
    data: list = field(default_factory=list)
    idx: int = 0
    module_idx: int = 0
    builder = flatbuffers.Builder(0)

    def save_to_flatbuff(self):
        Path(self.save_path).mkdir(exist_ok=True, parents=True)
        LayersStartLayersVector(self.builder, len(self.data))
        for i in reversed(self.data):
            self.builder.PrependUOffsetTRelative(i)
        layers = self.builder.EndVector()
        layer_name = self.builder.CreateString(self.name)
        LayerStart(self.builder)
        LayersAddLayers(self.builder, layers)
        LayersAddName(self.builder, layer_name)
        layers = LayersEnd(self.builder)
        self.builder.Finish(layers)
        buf = self.builder.Output()
        with open(f"{self.save_path}/{self.name}.data", "wb") as f:
            f.write(buf)

    @torch.no_grad()
    def parse_module(self, module, name: str):

        if isinstance(module, (nn.ModuleDict, nn.ModuleList, nn.Sequential)):
            for (new_name, new_module) in module.named_children():
                self.parse_module(new_module, name=f"{name}.{new_name}")
            return
        elif list(module.children()):
            module_name = module.__class__.__name__
            custom_module_name = f"{module_name}_{self.module_idx}"
            self.module_idx += 1
            for (new_name, new_module) in module.named_children():
                self.parse_module(new_module, name=f"{custom_module_name}.{new_name}")
            return
        elif isinstance(module, nn.ReLU):
            data = save_relu(module, builder=self.builder, name=name, idx=self.idx)
        elif isinstance(module, nn.AdaptiveAvgPool2d):
            data = save_adaptiveavgpool2d(
                module, builder=self.builder, name=name, idx=self.idx
            )
        elif isinstance(module, nn.Conv2d):
            data = save_conv2d(module, builder=self.builder, name=name, idx=self.idx)
        elif isinstance(module, nn.BatchNorm2d):
            data = save_batchnorm2d(
                module, builder=self.builder, name=name, idx=self.idx
            )
        elif isinstance(module, nn.Linear):
            data = save_linear(module, builder=self.builder, name=name, idx=self.idx)
        elif isinstance(module, nn.Dropout):
            data = save_dropout(module, builder=self.builder, name=name, idx=self.idx)
        elif isinstance(module, nn.MaxPool2d):
            data = save_maxpool2d(module, builder=self.builder, name=name, idx=self.idx)
        elif isinstance(module, nn.Flatten):
            data = save_flatten(module, builder=self.builder, name=name, idx=self.idx)
        else:
            raise NotImplementedError(type(module))
        self.idx += 1
        self.data.append(data)


from copy import deepcopy

parser = Parser(save_path="elo", name="conv2dsimple")
module = nn.Sequential(
    nn.Conv2d(3, 6, (1, 1), bias=False),
    nn.Conv2d(6, 3, (2, 2), bias=True),
    nn.BatchNorm2d(3),
    nn.MaxPool2d(kernel_size=2, stride=2),
    # nn.Flatten(start_dim=2),
    nn.AdaptiveAvgPool2d((24, 24)),
    nn.Flatten(),
    nn.Linear(in_features=24 * 24 * 3, out_features=3, bias=False),
)
inp = torch.rand(1, 3, 256, 256)

out = module(inp)
print(out)
# module = module.eval()
parser.parse_module(module=deepcopy(module), name="testconv")
parser.save_to_flatbuff()

torch.save(inp, "inp.pt")
torch.save(out, "out.pt")
