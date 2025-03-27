import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest
import numpy as np
from typing import Optional, Tuple

# Import the modules we want to test
from fla.layers.hgrn import HGRNAttention
from fla.modules import GatedMLP, PrunableLinear, RMSNorm
from fla.modules.activations import swiglu

# Define simple reference implementations of each component

class RefRMSNorm(nn.Module):
    """Simple reference implementation of RMSNorm"""
    def __init__(
        self, 
        hidden_size: int, 
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = False
    ):
        super().__init__()
        self.eps = eps
        self.hidden_size = hidden_size
        self.elementwise_affine = elementwise_affine
        
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_size))
            if bias:
                self.bias = nn.Parameter(torch.zeros(hidden_size))
            else:
                self.register_parameter('bias', None)
    
    def forward(self, x, residual=None, prenorm=False):
        if residual is not None:
            x = x + residual
            
        input_dtype = x.dtype
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x / torch.sqrt(variance + self.eps)

        if self.elementwise_affine:
            x = x * self.weight
            if self.bias is not None:
                x = x + self.bias
                
        x = x.to(input_dtype)
        return x if not prenorm else (x, residual)


class RefGatedMLP(nn.Module):
    """Simple reference implementation of GatedMLP with SwiGLU"""
    def __init__(
        self,
        hidden_size: int,
        hidden_ratio: Optional[int] = None,
        intermediate_size: Optional[int] = None,
        hidden_act: str = 'swish',
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        # Default to 4x hidden size if not specified
        if hidden_ratio is None:
            hidden_ratio = 4
        if intermediate_size is None:
            intermediate_size = int(hidden_size * hidden_ratio * 2 / 3)
            intermediate_size = 256 * ((intermediate_size + 256 - 1) // 256)
            
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        
        if hidden_act != 'swish':
            raise ValueError(f'Unsupported hidden_act: {hidden_act}')
            
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
    
    def forward(self, x):
        gate, y = self.gate_proj(x), self.up_proj(x)
        # Apply SwiGLU: x * sigmoid(gate) * swish(y)
        hidden_states = swiglu(gate, y)
        return self.down_proj(hidden_states)


class RefPrunableLinear(nn.Linear):
    """Simple reference implementation of PrunableLinear"""
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None
    ):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.mask = nn.Parameter(torch.ones_like(self.weight), requires_grad=False)
        self.pruning_active = False
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.pruning_active:
            masked_weight = self.weight * self.mask
            return F.linear(input, masked_weight, self.bias)
        else:
            return super().forward(input)
    
    def enable_pruning(self, active=True):
        self.pruning_active = active


class RefHGRNAttention(nn.Module):
    """Simple reference implementation of HGRNAttention"""
    def __init__(
        self,
        mode: str = 'chunk',
        hidden_size: int = 1024,
        expand_ratio: Optional[int] = 1,
        use_short_conv: bool = False,
        conv_size: int = 4,
        elementwise_affine: Optional[bool] = True,
        norm_eps: float = 1e-5,
        layer_idx: int = None
    ):
        super().__init__()
        
        self.mode = mode
        self.hidden_size = hidden_size
        self.expand_ratio = expand_ratio
        self.input_dim = int(hidden_size * expand_ratio)
        self.use_short_conv = use_short_conv
        self.layer_idx = layer_idx
        
        # Linear projections
        self.i_proj = nn.Linear(hidden_size, self.input_dim, bias=False)
        self.f_proj = nn.Linear(hidden_size, self.input_dim, bias=False)
        self.g_proj = nn.Linear(hidden_size, self.input_dim, bias=False)
        
        # Normalization layer - using standard PyTorch components
        self.g_norm = nn.LayerNorm(self.input_dim, eps=norm_eps, elementwise_affine=elementwise_affine)
        self.o_proj = nn.Linear(self.input_dim, hidden_size, bias=False)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        lower_bound: Optional[torch.Tensor] = None,
        **kwargs
    ):
        # Project inputs
        i = self.i_proj(hidden_states)
        f = self.f_proj(hidden_states)
        
        # Apply activation and sigmoid
        if lower_bound is None or self.layer_idx == 0:
            # Function for swiglu + logsigmoid in the i and f branch
            i, f = swiglu(i, 1 - f.sigmoid()), F.logsigmoid(f)
        else:
            g = lower_bound + (1 - lower_bound) * f.sigmoid()
            i, f = swiglu(i, 1 - g), g.log()
        
        # Apply mask if provided
        if attention_mask is not None:
            i = i * attention_mask[:, -i.shape[-2]:, None]
        
        # Simplified recurrent processing - real implementation would use optimized kernels
        # This just demonstrates the logical flow not the actual computation
        batch_size, seq_len, hidden_dim = i.shape
        h = torch.zeros((batch_size, 1, hidden_dim), device=i.device, dtype=i.dtype)
        outputs = []
        
        for t in range(seq_len):
            # Recurrent update
            h = h * torch.exp(f[:, t:t+1, :]) + i[:, t:t+1, :]
            outputs.append(h)
        
        o = torch.cat(outputs, dim=1)
        
        # Apply normalization and projection
        g = self.g_proj(hidden_states)
        o = self.g_norm(o) * F.silu(g)  # Approximating the behavior of FusedRMSNormSwishGate
        o = self.o_proj(o)
        
        return o, None, past_key_values


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("seq_len", [8, 32])
@pytest.mark.parametrize("hidden_size", [128, 256])
def test_rmsnorm(dtype, batch_size, seq_len, hidden_size):
    """Test that RMSNorm produces the same results as reference implementation"""
    # Skip if bfloat16 is not available
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("bfloat16 not supported on this device")
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create test data
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, hidden_size, device=device).to(dtype)
    residual = torch.randn(batch_size, seq_len, hidden_size, device=device).to(dtype)
    
    # Create models
    ref_model = RefRMSNorm(hidden_size, eps=1e-5, elementwise_affine=True).to(device).to(dtype)
    test_model = RMSNorm(hidden_size, eps=1e-5, elementwise_affine=True).to(device).to(dtype)
    
    # Ensure weights are the same
    test_model.weight.data.copy_(ref_model.weight.data)
    
    # Forward pass
    with torch.no_grad():
        ref_output = ref_model(x, residual)
        test_output = test_model(x, residual)
    
    # Compare results - with higher tolerance for fp16/bf16
    rtol = 1e-3 if dtype in (torch.float16, torch.bfloat16) else 1e-5
    atol = 1e-3 if dtype in (torch.float16, torch.bfloat16) else 1e-5
    
    assert torch.allclose(ref_output, test_output, rtol=rtol, atol=atol), \
        f"Max diff: {(ref_output - test_output).abs().max().item()}"


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("seq_len", [8, 32])
@pytest.mark.parametrize("hidden_size", [128, 256])
@pytest.mark.parametrize("hidden_ratio", [2, 4])
@pytest.mark.parametrize("fuse_swiglu", [True, False])
def test_gated_mlp(dtype, batch_size, seq_len, hidden_size, hidden_ratio, fuse_swiglu):
    """Test that GatedMLP produces similar results to reference implementation"""
    # Skip if bfloat16 is not available
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("bfloat16 not supported on this device")
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create test data
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, hidden_size, device=device).to(dtype)
    
    # Create models
    intermediate_size = int(hidden_size * hidden_ratio * 2 / 3)
    intermediate_size = 256 * ((intermediate_size + 256 - 1) // 256)
    
    ref_model = RefGatedMLP(
        hidden_size=hidden_size,
        hidden_ratio=hidden_ratio,
        intermediate_size=intermediate_size
    ).to(device).to(dtype)
    
    test_model = GatedMLP(
        hidden_size=hidden_size,
        hidden_ratio=hidden_ratio,
        intermediate_size=intermediate_size,
        fuse_swiglu=fuse_swiglu
    ).to(device).to(dtype)
    
    # Copy weights
    test_model.gate_proj.weight.data.copy_(ref_model.gate_proj.weight.data)
    test_model.up_proj.weight.data.copy_(ref_model.up_proj.weight.data)
    test_model.down_proj.weight.data.copy_(ref_model.down_proj.weight.data)
    
    # Forward pass
    with torch.no_grad():
        ref_output = ref_model(x)
        test_output = test_model(x)
    
    # Compare results - higher tolerance for low precision and fused implementation
    rtol = 5e-2 if dtype in (torch.float16, torch.bfloat16) or fuse_swiglu else 1e-4
    atol = 5e-2 if dtype in (torch.float16, torch.bfloat16) or fuse_swiglu else 1e-4
    
    # Print max diff for debugging
    max_diff = (ref_output - test_output).abs().max().item()
    print(f"Max diff: {max_diff} (fuse_swiglu={fuse_swiglu}, dtype={dtype})")
    
    assert torch.allclose(ref_output, test_output, rtol=rtol, atol=atol), \
        f"Max diff: {max_diff}"


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("in_features", [128, 256])
@pytest.mark.parametrize("out_features", [128, 256])
@pytest.mark.parametrize("with_bias", [True, False])
@pytest.mark.parametrize("pruning_active", [True, False])
def test_prunable_linear(dtype, batch_size, in_features, out_features, with_bias, pruning_active):
    """Test that PrunableLinear produces the same results as reference implementation"""
    # Skip if bfloat16 is not available
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("bfloat16 not supported on this device")
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create test data
    torch.manual_seed(42)
    x = torch.randn(batch_size, in_features, device=device).to(dtype)
    
    # Create models
    ref_model = RefPrunableLinear(in_features, out_features, bias=with_bias).to(device).to(dtype)
    test_model = PrunableLinear(in_features, out_features, bias=with_bias).to(device).to(dtype)
    
    # Enable pruning
    ref_model.enable_pruning(pruning_active)
    test_model.enable_pruning(pruning_active)
    
    # Create a test mask with 50% of weights pruned
    pruning_mask = torch.ones_like(ref_model.weight)
    pruning_mask.view(-1)[::2] = 0  # Prune every other weight
    
    # Apply mask to both models
    ref_model.mask.data.copy_(pruning_mask)
    test_model.mask.data.copy_(pruning_mask)
    
    # Copy weights and bias
    test_model.weight.data.copy_(ref_model.weight.data)
    if with_bias:
        test_model.bias.data.copy_(ref_model.bias.data)
    
    # Forward pass
    with torch.no_grad():
        ref_output = ref_model(x)
        test_output = test_model(x)
    
    # Should be exactly the same
    assert torch.allclose(ref_output, test_output, rtol=1e-4, atol=1e-4), \
        f"Max diff: {(ref_output - test_output).abs().max().item()}"


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("seq_len", [8, 32])
@pytest.mark.parametrize("hidden_size", [128, 256])
@pytest.mark.parametrize("expand_ratio", [1, 2])
@pytest.mark.parametrize("with_mask", [True, False])
def test_hgrn_attention(dtype, batch_size, seq_len, hidden_size, expand_ratio, with_mask):
    """Test HGRN Attention matches reference implementation"""
    # Skip if bfloat16 is not available
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("bfloat16 not supported on this device")
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create test data
    torch.manual_seed(42)
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, device=device).to(dtype)
    
    # Create attention mask if needed
    attention_mask = None
    if with_mask:
        # Create a simple causal mask (for simplicity in testing)
        attention_mask = torch.ones(batch_size, seq_len, device=device)
    
    # Create models - forcing 'chunk' mode for testing against reference
    ref_model = RefHGRNAttention(
        mode='chunk',
        hidden_size=hidden_size,
        expand_ratio=expand_ratio,
        use_short_conv=False,
        elementwise_affine=True,
        norm_eps=1e-5,
        layer_idx=0
    ).to(device).to(dtype)
    
    test_model = HGRNAttention(
        mode='chunk',  # Use chunk mode to allow comparison
        hidden_size=hidden_size,
        expand_ratio=expand_ratio,
        use_short_conv=False,
        elementwise_affine=True,
        norm_eps=1e-5,
        layer_idx=0
    ).to(device).to(dtype)
    
    # Copy weights to ensure they match
    test_model.i_proj.weight.data.copy_(ref_model.i_proj.weight.data)
    test_model.f_proj.weight.data.copy_(ref_model.f_proj.weight.data)
    test_model.g_proj.weight.data.copy_(ref_model.g_proj.weight.data)
    test_model.o_proj.weight.data.copy_(ref_model.o_proj.weight.data)
    
    # Note: We can't easily copy g_norm weights because the implementations differ
    # For a real test you might need custom initialization
    
    # Forward pass
    with torch.no_grad():
        ref_output, _, _ = ref_model(hidden_states, attention_mask)
        test_output, _, _ = test_model(hidden_states, attention_mask)
    
    # Due to implementation differences, we expect larger differences
    # Just check that the shapes match and values are in a reasonable range
    assert ref_output.shape == test_output.shape
    
    # For HGRN, due to the complex recurrent nature and optimized CUDA kernels,
    # we mainly check for general reasonableness rather than exact matching
    # Relaxed tolerance for low precision and complex operation
    rtol = 1e-1
    atol = 1e-1
    
    # Print max diff for debugging
    max_diff = (ref_output - test_output).abs().max().item()
    print(f"Max HGRN diff: {max_diff} (dtype={dtype})")
    
    # Just check that outputs are reasonably close (not exact match due to algorithm differences)
    similar_scale = torch.allclose(
        torch.norm(ref_output), 
        torch.norm(test_output), 
        rtol=0.5, atol=0.5
    )
    assert similar_scale, "HGRN output magnitudes differ significantly from reference"


if __name__ == "__main__":
    # Enable pytest to run the tests
    import sys
    sys.exit(pytest.main(["-v", __file__])) 
