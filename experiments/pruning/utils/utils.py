import torch
from typing import Tuple
import torch.nn as nn
from collections import OrderedDict
from torch.nn.utils import parametrize


def remove_parametrizations(model):
    """Function to remove parameterizations from a model."""

    parametrized_modules = {}

    for name, module in model.named_modules():
        if hasattr(module, "parametrizations"):
            parametrized_modules[name] = []

            for p_name in list(module.parametrizations.keys()):
                orig_parameter = getattr(module.parametrizations, p_name)
                orig_parameter = orig_parameter.original.data.detach().clone()
                parametrized_modules[name].append(
                    (p_name, module.parametrizations[p_name], orig_parameter)
                )
                parametrize.remove_parametrizations(module, p_name, True)
    
    return parametrized_modules


def get_attr_by_name(module, name):
    """ """
    for s in name.split("."):
        module = getattr(module, s)

    return module


def get_parent_name(qualname: str) -> Tuple[str, str]:
    """
    Splits a ``qualname`` into parent path and last atom.
    For example, `foo.bar.baz` -> (`foo.bar`, `baz`)
    """
    *parent, name = qualname.rsplit(".", 1)
    return parent[0] if parent else "", name


def get_parent_module(module, attr_path):
    """
    Returns parent module of module.attr_path.

    Parameters
    ----------
    module: torch.nn.Module.
    attr_path: str.
    """
    parent_name, _ = get_parent_name(attr_path)

    if parent_name != "":
        parent = get_attr_by_name(module, parent_name)
    else:
        parent = module

    return parent


def remove_all_hooks(model: torch.nn.Module) -> None:
    """ """
    for name, child in model._modules.items():
        if child is not None:
            if hasattr(child, "_forward_hooks"):
                child._forward_hooks = OrderedDict()
            remove_all_hooks(child)


def fuse_batchnorm(
    model, 
    fx_model=None, 
    convs=None
):
    """
    Fuse conv and bn only if conv is in convs argument.

    Parameters
    ----------
    model: torch.nn.Module.
    fx_model: torch.fx.GraphModule.
    convs: List[torch.nn.ConvNd].
    """
    if fx_model is None:
        fx_model: torch.fx.GraphModule = torch.fx.symbolic_trace(model)
    modules = dict(fx_model.named_modules())

    for node in fx_model.graph.nodes:
        if node.op != "call_module":
            continue
        if (
            type(modules[node.target]) is nn.BatchNorm2d
            and type(modules[node.args[0].target]) is nn.Conv2d
        ):
            to_fuse = True if convs is None else node.args[0].target in convs 
            if to_fuse:
                if len(node.args[0].users) > 1:
                    continue
                conv = modules[node.args[0].target]
                bn = modules[node.target]
                _inplace_conv_bn_fusion(conv, bn)
                parent_name, attr_name = get_parent_name(node.target)
                parent = get_parent_module(model, node.target)
                setattr(parent, attr_name, torch.nn.Identity())


def _inplace_conv_bn_fusion(conv, bn):
    """ """
    assert not (conv.training or bn.training), "Fusion only for eval!"
    conv.weight.data, bias = _fuse_conv_bn_weights(
        conv.weight,
        conv.bias,
        bn.running_mean,
        bn.running_var,
        bn.eps,
        bn.weight,
        bn.bias,
    )

    if conv.bias is None:
        conv.bias = torch.nn.Parameter(bias).to(conv.weight.device)
    else:
        conv.bias.data = bias


def _fuse_conv_bn_weights(conv_w, conv_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b):
    """ """
    if conv_b is None:
        conv_b = torch.zeros_like(bn_rm)
    if bn_w is None:
        bn_w = torch.ones_like(bn_rm)
    if bn_b is None:
        bn_b = torch.zeros_like(bn_rm)
    bn_var_rsqrt = torch.rsqrt(bn_rv + bn_eps)

    conv_w = conv_w * (bn_w * bn_var_rsqrt).reshape(
        [-1] + [1] * (len(conv_w.shape) - 1)
    )
    conv_b = (conv_b - bn_rm) * bn_var_rsqrt * bn_w + bn_b

    return conv_w, conv_b


def run_model(model, inputs, req_grad=False):
    if isinstance(inputs, torch.Tensor):
        inputs = (inputs, )
    with torch.set_grad_enabled(mode=req_grad):
        outputs = model(*inputs)
    
    return outputs