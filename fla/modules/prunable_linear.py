import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class StraightThroughEstimator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, mask):
        # Apply mask in forward pass
        output = input * mask
        # Save mask for backward
        ctx.save_for_backward(mask)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        # Get saved mask
        mask, = ctx.saved_tensors
        # Pass gradients through, ignoring the mask
        return grad_output, None


class PrunableLinear(nn.Linear):
    """
    A prunable version of nn.Linear that supports weight pruning with a binary mask.
    Uses straight-through estimator during training to allow gradients to flow
    through masked weights.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None
    ):
        super().__init__(in_features, out_features, bias, device, dtype)
        
        # Initialize pruning mask
        self.register_buffer('mask', torch.ones_like(self.weight))
        self.pruning_active = False
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.pruning_active:
            # Apply mask with STE in forward pass
            masked_weight = StraightThroughEstimator.apply(self.weight, self.mask)
            return F.linear(input, masked_weight, self.bias)
        else:
            return super().forward(input)
    
    def enable_pruning(self, active=True):
        self.pruning_active = active
        
    def get_pruning_state(self):
        """Return the current pruning state for analysis"""
        sparsity = (self.mask == 0).float().mean().item()
        return {
            "sparsity": sparsity,
            "pruned_params": (self.mask == 0).sum().item(),
            "total_params": self.mask.numel()
        } 
