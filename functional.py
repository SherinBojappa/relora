import math
import copy

import torch
import torch.nn as nn
import torch.nn.init
import torch.nn.functional as F

from megatron import print_rank_0

from megatron.relora import ReLoRaLinear


def merge_and_reinit_functional(module):
    if not isinstance(module, ReLoRaLinear):
        return

    _delta = module.lora_B.weight @ module.lora_A.weight
    module.weight.data += _delta * module._post_lora_scale()
    nn.init.kaiming_uniform_(module.lora_A.weight, a=math.sqrt(5))

    nn.init.zeros_(module.lora_B.weight)
    if module.trainable_scaling:
        nn.init.zeros_(module.scaling)


def wrap_with_ReLoRa(model, r=128, lora_alpha=32, lora_dropout=0.1, trainable_scaling=False):
    if r <= 0:
        raise ValueError("r must be positive. If you want r == 0, use the original model.")

    new_model = model

    for module_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if isinstance(module, ReLoRaLinear):
            print_rank_0("WARNING: Trying to wrap ReLoRA into ReLoRA. Are you sure this is what you want?")
            continue

        new_module = ReLoRaLinear(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            trainable_scaling=trainable_scaling,
        )

        new_module.weight.data = module.weight.data
        if module.bias is not None:
            new_module.bias.data = module.bias.data
        nn.init.zeros_(new_module.lora_A.weight)

        parent = _get_parent(module_name, new_model)
        module_suffix = module_name.split(".")[-1]
        setattr(parent, module_suffix, new_module)

    return new_model


def _get_parent(module_name, model):
    module_names_list = module_name.split(".")
    parent_name = ".".join(module_names_list[:-1])
    parent = model.get_submodule(parent_name)
    return parent


def merge_and_reinit(model):
    for module in model.modules():
        if isinstance(module, ReLoRaLinear):
            merge_and_reinit_functional(module)


def merge_and_unwrap(model) -> nn.Module:
    unwrapped_model = copy.deepcopy(model) # Create a deep copy of the model
    for module_name, module in model.named_modules():
        if isinstance(module, ReLoRaLinear):
            new_module = nn.Linear(module.in_features, module.out_features, bias=(module.bias is not None))
            merge_and_reinit_functional(module)
            new_module.weight.data = module.weight.data
            if module.bias is not None:
                new_module.bias.data = module.bias.data
                
            parent = _get_parent(module_name, unwrapped_model)
            module_suffix = module_name.split(".")[-1]
            setattr(parent, module_suffix, new_module) 

    return unwrapped_model
