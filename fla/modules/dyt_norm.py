# -*- coding: utf-8 -*-
# Implementation of DyT (Dynamic Tanh) normalization module 

import torch
import torch.nn as nn


class DyT(nn.Module):
    """
    Dynamic Tanh normalization module.
    
    Args:
        num_features: Size of the feature dimension
        alpha_init_value: Initial value for the alpha parameter (default: 0.5)
        bias: Whether to use bias (default: True)
    """
    def __init__(self, num_features, alpha_init_value=0.5, bias=True):
        super().__init__()
        self.num_features = num_features
        self.alpha_init_value = alpha_init_value
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.register_parameter("bias", None)
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(num_features))
    
    def forward(self, x, residual=None, prenorm=False, residual_in_fp32=False):
        """
        Forward pass.
        
        Args:
            x: Input tensor
            residual: Optional residual tensor to add to input
            prenorm: Whether to return the pre-normalized input as well
            residual_in_fp32: Whether to keep the residual in float32
            
        Returns:
            Normalized tensor, or tuple of normalized tensor and pre-normalized input
        """
        # Handle input shape
        x_shape_og = x.shape
        x = x.reshape(-1, x.shape[-1])
        
        # Add residual if provided
        if residual is not None:
            residual = residual.reshape(-1, residual.shape[-1])
            x = x + residual
            
        # Store pre-normalized tensor if needed
        if prenorm:
            x_pre = x.clone()
            
        # Apply dynamic tanh normalization
        x = torch.tanh(self.alpha * x)
        x = x * self.weight
        
        if self.bias is not None:
            x = x + self.bias
            
        # Restore original shape
        x = x.reshape(x_shape_og)
        
        if prenorm:
            x_pre = x_pre.reshape(x_shape_og)
            return x, x_pre
        return x
    
    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}({self.num_features}"
        s += f", alpha_init_value={self.alpha_init_value}"
        s += f", bias={self.bias is not None}"
        s += ")"
        return s


class FusedDyTSwishGate(nn.Module):
    """
    Fused DyT and Swish Gate module for HGRN architecture.
    
    Args:
        hidden_size: Size of the feature dimension
        alpha_init_value: Initial value for the alpha parameter (default: 0.5)
        bias: Whether to use bias (default: True)
    """
    def __init__(
        self,
        hidden_size,
        alpha_init_value=0.5,
        bias=True
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.alpha_init_value = alpha_init_value
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.register_parameter("bias", None)
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(hidden_size))

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}({self.hidden_size}"
        s += f", alpha_init_value={self.alpha_init_value}"
        s += f", bias={self.bias is not None}"
        s += ")"
        return s

    def forward(self, x, o, residual=None, prenorm=False, residual_in_fp32=False):
        """
        Forward pass with gate.
        
        Args:
            x: Input tensor
            o: Gate tensor
            residual: Optional residual tensor to add to input
            prenorm: Whether to return the pre-normalized input as well
            residual_in_fp32: Whether to keep the residual in float32
            
        Returns:
            Gated normalized tensor, or tuple of gated normalized tensor and pre-normalized input
        """
        # Handle input shape
        x_shape_og = x.shape
        o_shape_og = o.shape
        x = x.reshape(-1, x.shape[-1])
        o = o.reshape(-1, o.shape[-1])
        
        # Add residual if provided
        if residual is not None:
            residual = residual.reshape(-1, residual.shape[-1])
            x = x + residual
            
        # Store pre-normalized tensor if needed
        if prenorm:
            x_pre = x.clone()
            
        # Apply dynamic tanh normalization
        x = torch.tanh(self.alpha * x)
        x = x * self.weight
        
        if self.bias is not None:
            x = x + self.bias
            
        # Apply swish gate
        x = x * o * torch.sigmoid(o)
        
        # Restore original shape
        x = x.reshape(x_shape_og)
        
        if prenorm:
            x_pre = x_pre.reshape(x_shape_og)
            return x, x_pre
        return x
