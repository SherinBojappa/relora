import math
from functools import partial

import torch
from torch.optim.lr_scheduler import LambdaLR

#from megatron import print_rank_0


def get_ragged_cosine_schedule(
    optimizer,
    *,
    num_training_steps,
    first_warmup_steps,
    restart_warmup_steps,
    reset_freq,
    min_lr_ratio=0.1,
    adjust_step=0,
    last_epoch=-1,
):
    if reset_freq is None:
        raise ValueError("reset_freq must be specified for ragged_cosine_schedule")

    if num_training_steps % reset_freq != 0:
        raise ValueError(f"num_training_steps ({num_training_steps}) must be divisible by reset_freq ({reset_freq})")

    lr_lambda = partial(
        _get_cosine_schedule_with_multiple_warmups_lambda,
        num_training_steps=num_training_steps,
        first_warmup_steps=first_warmup_steps,
        restart_warmup_steps=restart_warmup_steps,
        reset_freq=reset_freq,
        min_lr_ratio=min_lr_ratio,
        adjust_step=adjust_step,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def _get_cosine_schedule_with_multiple_warmups_lambda(
    current_step,
    *,
    num_training_steps,
    first_warmup_steps,
    restart_warmup_steps,
    reset_freq,
    min_lr_ratio,
    adjust_step,
):
    """
    Args:
        adjust_step: useful when continuing training from a warmed up checkpoint,
            it allows to sync the resets by reducing the number of steps
            after the first warmup and before the first reset.
            Thus, your ReLoRA resets can be synced with the optimizer resets.
    """
    assert 0 < min_lr_ratio <= 1.0, "min_lr_ratio must be in (0,1]"
    assert reset_freq > 0, "reset_freq must be positive"
    assert adjust_step + first_warmup_steps <= num_training_steps, "warmup + adjust_step is more than full training steps"
    assert adjust_step + first_warmup_steps <= reset_freq, "the first reset will happen before the warmup is done"

    if current_step < first_warmup_steps:
        return float(current_step) / float(max(1, first_warmup_steps))

    _current_step = current_step + adjust_step

    restart_step = _current_step % reset_freq
    restart_number = _current_step // reset_freq

    if restart_step < restart_warmup_steps:
        # get expected lr multipler at the end of the warmup
        end_of_warmup_progress = (
            float(restart_number * reset_freq) /
            float(max(1, num_training_steps - first_warmup_steps))
        )

        _cosine_decay = 0.5 * (1.0 + math.cos(math.pi * end_of_warmup_progress))
        warmup_lr_multiplier = min_lr_ratio + (1.0 - min_lr_ratio) * _cosine_decay

        return float(restart_step) / float(max(1, restart_warmup_steps)) * warmup_lr_multiplier

    progress = float(_current_step - first_warmup_steps) / float(max(1, num_training_steps - first_warmup_steps))
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))

    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay


@torch.no_grad()
def magnitude_pruning_(tensor, prune_ratio):
    """
    Performs magnitude pruning dimensionality reduction **inplace**.
    Only reduces the inner dimensionality, does not affect the shape of the tensor
    """
    tensor_magnitude = torch.abs(tensor)
    threshold = torch.quantile(tensor_magnitude.flatten().to(dtype=torch.float32), prune_ratio).to(dtype=tensor.dtype)

    mask = tensor_magnitude > threshold
    tensor.mul_(mask.to(dtype=tensor.dtype))


def reset_optimizer(
    optimizer,
    *,
    reset_params: list[torch.nn.Parameter],
    pruning_amount: float,
):
    if not(0 <= pruning_amount <= 1):
        raise RuntimeError("pruning_amount must be in [0,1]")

    pruning_fn = partial(magnitude_pruning_, prune_ratio=pruning_amount)

    # ############################################################
    # A reminder on how optimizer state is structured for regular optimizers:
    # optimizer.state is a dict[torch.nn.Parameter, dict[str, torch.Tensor]]
    # optimizer.state[p] is a dict[str, torch.Tensor] where str is
    # an optimizer state key e.g., "exp_avg", "exp_avg_sq"
    # Note that none of these tensors has parameter names
    # and parameter maps to a **dictionary** of opt. states, not a tensor
    # ############################################################
    n_zeros = 0
    n_total = 0

    from torch.optim.optimizer import Optimizer

    optimizer_state = optimizer.state

    for p in reset_params:
        param_state = optimizer_state[p]
        if len(param_state) == 0: # no state for this param, happens for ZeRo optimizer
            continue
        for key, state_for_key in param_state.items():
            if not isinstance(state_for_key, torch.Tensor):
                continue
            pruning_fn(param_state[key])  # pruning fn has to be inplace to keep the same keys in the dict
            n_total += param_state[key].numel()
            n_zeros += torch.sum(param_state[key] == 0).item()

    _zeroed = n_zeros / (1e-7 + n_total) * 100
    #print_rank_0(f"Percent of optimizer states zeroed: {_zeroed:.2f}")
    print(f"Percent of optimizer states zeroed: {_zeroed:.2f}")
