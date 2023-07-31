import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from megatron import print_rank_0


class ReLoRaModel(torch.nn.Module):
    def __init__(
        self,
        model,
        *,
        r=128,
        lora_alpha=32,
        lora_dropout=0.1,
        trainable_scaling=False,
    ):
        if r <= 0:
            raise ValueError("r must be positive. If you want r == 0, use the original model.")

        super().__init__()
        self.wrapped_model: nn.Module = model
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.trainable_scaling = trainable_scaling

        # patch methods
        self.forward = self.wrapped_model.forward

        for module_name, module in self.wrapped_model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if isinstance(module, ReLoRaLinear):
                print_rank_0("WARNING: Trying to wrap ReLoRA into ReLoRA. Are you sure this is what you want?")
                continue

            new_module = ReLoRaLinear(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                r=self.r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                lora_only=self.lora_only,
                trainable_scaling=self.trainable_scaling,
            )

            new_module.weight.data = module.weight.data
            if module.bias is not None:
                new_module.bias.data = module.bias.data
            # make lora'ed network to be exacty the same as the original network at initialization
            nn.init.zeros_(new_module.lora_A.weight)
            assert new_module.lora_A.bias is None
            assert new_module.lora_B.bias is None

            parent = self._get_parent(module_name)
            module_suffix = module_name.split(".")[-1]
            setattr(parent, module_suffix, new_module)

    def _get_parent(self, module_name):
        module_names_list = module_name.split(".")
        parent_name = ".".join(module_names_list[:-1])
        parent = self.wrapped_model.get_submodule(parent_name)
        return parent

    def merge_and_reinit(self):
        for module in self.modules():
            if isinstance(module, ReLoRaLinear):
                module.merge_and_reinit()

    def merge_and_unwrap(self) -> nn.Module:
        self.merge_and_reinit()
        return self.wrapped_model

    def save_pretrained(self, path):
        raise RuntimeError("Call .merge_and_unwrap() and save the unwrapped model instead")

    def state_dict(self):
        raise RuntimeError("Call .merge_and_unwrap() and save the unwrapped model instead")

    def load_state_dict(self, state_dict):
        raise RuntimeError("Call .merge_and_unwrap() and load the unwrapped model instead")


# The code is based on https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
class ReLoRaLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int,
        lora_alpha: int = 1,
        lora_dropout: float = 0.1,
        trainable_scaling: bool = False,
        **kwargs,
    ):
        """Wraps linear layer x W into x W + x W_a @ W_b * lora_alpha / r
        
        Notice that scale = lora_alpha / r.
        """
        if r <= 0:
            raise ValueError("r must be positive. If you want r == 0, use the original model.")

        nn.Linear.__init__(self, in_features, out_features, **kwargs)

        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(p=lora_dropout)
        self.trainable_scaling = trainable_scaling

        if r > 0:
            self.lora_A = nn.Linear(in_features, r, bias=False)
            self.lora_B = nn.Linear(r, out_features, bias=False)
            if trainable_scaling:
                self.scaling = nn.Parameter(torch.tensor([1.]), requires_grad=True)
            else:
                self.scaling = self.lora_alpha / self.r

            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        if not hasattr(self, "lora_A"):
            # we are in nn.Linear calling reset_parameters
            nn.Linear.reset_parameters(self)
            return

        if not self.lora_only:
            nn.init.zeros_(self.weight)
            if self.bias is not None:
                nn.init.zeros_(self.bias)

        # disgard original, but now we need to init both A and B with kaiming
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_B.weight, a=math.sqrt(5))
    
    def _post_lora_scale(self):
        if self.trainable_scaling:
            return self.scaling.tanh()

        return self.scaling

    @torch.no_grad()
    def merge_and_reinit(self):
        if self.lora_only:
            print("WARNING: Skipping merge and reinit, because only lora parameters are used")
            return

        self.weight.data += self.lora_B.weight @ self.lora_A.weight * self._post_lora_scale()
        self.merged = False
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))

        nn.init.zeros_(self.lora_B.weight)
        if self.trainable_scaling:
            nn.init.zeros_(self.scaling)

    def forward(self, x: torch.Tensor):
        if self.lora_only:
            # just lora
            return self.lora_B(self.lora_A(self.lora_dropout(x))) * self._post_lora_scale()

        result = F.linear(x, self.weight, bias=self.bias)

        if self.r > 0:
            result += self.lora_B(self.lora_A(self.lora_dropout(x))) * self._post_lora_scale()
        return result
