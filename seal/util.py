import torch
import casadi as ca
import numpy as np


def find_idx_for_labels(sub_vars: ca.SX, sub_label: str) -> list[int]:
    """Return a list of indices where sub_label is part of the variable label."""
    return [
        idx
        for idx, label in enumerate(sub_vars.str().strip("[]").split(", "))
        if sub_label in label
    ]


def tensor_to_numpy(tensor: torch.Tensor):
    return tensor.detach().cpu().numpy()
