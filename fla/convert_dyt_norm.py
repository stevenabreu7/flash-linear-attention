# -*- coding: utf-8 -*-
# Implementation of DyT (Dynamic Tanh) normalization for HGRN models
# Reference implementation below from https://github.com/jiachenzhu/DyT
"""
import torch
import torch.nn as nn
from timm.layers import LayerNorm2d


class DynamicTanh(nn.Module):
    def __init__(self, normalized_shape, channels_last, alpha_init_value=0.5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.alpha_init_value = alpha_init_value
        self.channels_last = channels_last

        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        if self.channels_last:
            x = x * self.weight + self.bias
        else:
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
        return x

    def extra_repr(self):
        return f"normalized_shape={self.normalized_shape}, alpha_init_value={self.alpha_init_value}, channels_last={self.channels_last}"


def convert_ln_to_dyt(module):
    module_output = module
    if isinstance(module, nn.LayerNorm):
        module_output = DynamicTanh(module.normalized_shape, not isinstance(module, LayerNorm2d))
    for name, child in module.named_children():
        module_output.add_module(name, convert_ln_to_dyt(child))
    del module
    return module_output
"""

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM
from fla.modules import DyT, FusedDyTSwishGate


def convert_rmsnorm_to_dyt(module, alpha_init_value=0.5):
    """
    Recursively replaces all RMSNorm layers in a module with DyT layers.
    Also replaces FusedRMSNormSwishGate with FusedDyTSwishGate.
    
    Args:
        module: The module to modify
        alpha_init_value: Initial value for the alpha parameter (default: 0.5)
        
    Returns:
        Modified module with DyT normalization instead of RMSNorm
    """
    from fla.modules import RMSNorm, FusedRMSNormSwishGate
    
    # Create a new module to avoid modifying the original
    if isinstance(module, RMSNorm) or isinstance(module, nn.RMSNorm):
        # Transfer parameters from RMSNorm to DyT
        dyt = DyT(
            num_features=module.weight.shape[0],
            alpha_init_value=alpha_init_value,
            bias=True  # DyT uses bias by default
        )
        
        # Copy weights (scale) from RMSNorm
        dyt.weight.data.copy_(module.weight.data)
        
        # Return the new DyT module
        return dyt
    
    # Handle the special case of FusedRMSNormSwishGate
    elif isinstance(module, FusedRMSNormSwishGate):
        dyt_gate = FusedDyTSwishGate(
            hidden_size=module.weight.shape[0],
            alpha_init_value=alpha_init_value,
            bias=True
        )
        
        # Copy weights from the gate
        dyt_gate.weight.data.copy_(module.weight.data)
        
        # Return the new DyT gate module
        return dyt_gate
    
    # Recursively process children
    for name, child in list(module.named_children()):
        new_child = convert_rmsnorm_to_dyt(child, alpha_init_value)
        if new_child is not child:  # Only set if changed
            setattr(module, name, new_child)
    
    return module


def create_dyt_model(config_path, model_path=None, checkpoint_path=None, alpha_init_value=0.5):
    """
    Creates a model with DyT normalization instead of RMSNorm.
    
    Args:
        config_path: Path to the model configuration
        model_path: Optional path to load model weights from HF format
        checkpoint_path: Optional path to load model weights from checkpoint
        alpha_init_value: Initial value for the alpha parameter (default: 0.5)
        
    Returns:
        Model with DyT normalization
    """
    # Load configuration
    config = AutoConfig.from_pretrained(config_path)
    
    # Create model
    model = AutoModelForCausalLM.from_config(config)
    
    # Load weights if specified
    if model_path:
        model = AutoModelForCausalLM.from_pretrained(model_path)
    elif checkpoint_path:
        # Load checkpoint weights
        # This would need to be implemented based on your checkpoint loading logic
        pass
    
    # Convert RMSNorm to DyT
    model = convert_rmsnorm_to_dyt(model, alpha_init_value)
    
    return model


# Example usage:
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser("Convert HGRN model to use DyT normalization")
    parser.add_argument("--config", type=str, required=True, help="Path to model config file")
    parser.add_argument("--model", type=str, default=None, help="Path to model weights (HF format)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--alpha", type=float, default=0.5, help="Initial value for DyT alpha parameter")
    parser.add_argument("--output", type=str, default=None, help="Output path to save the converted model")
    
    args = parser.parse_args()
    
    # Create model with DyT normalization
    model = create_dyt_model(args.config, args.model, args.checkpoint, args.alpha)
    
    # Save model if output specified
    if args.output:
        model.save_pretrained(args.output)
        print(f"Model saved to {args.output}")
    
    print("Conversion complete!")
