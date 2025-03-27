import pytest
import torch

@pytest.fixture(scope="session")
def cuda_device():
    """Return a CUDA device if available, else CPU"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture(scope="session")
def supports_bf16():
    """Check if the device supports bfloat16"""
    if not torch.cuda.is_available():
        return False
    return torch.cuda.is_bf16_supported()

@pytest.fixture
def setup_seed():
    """Set a fixed seed for reproducibility"""
    seed = 42
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed

@pytest.fixture(params=[torch.float16, torch.bfloat16])
def mixed_precision_dtype(request, supports_bf16):
    """Provide mixed precision dtype, skipping bfloat16 if not supported"""
    if request.param == torch.bfloat16 and not supports_bf16:
        pytest.skip("bfloat16 not supported on this device")
    return request.param

@pytest.fixture
def compare_tensor_values():
    """Return a function to compare tensor values with appropriate tolerance based on dtype"""
    def _compare(x, y, dtype):
        # Adjust tolerance based on precision
        if dtype in (torch.float16, torch.bfloat16):
            rtol, atol = 1e-2, 1e-2
        else:
            rtol, atol = 1e-5, 1e-5
            
        # Check if tensors are close
        max_diff = (x - y).abs().max().item()
        are_close = torch.allclose(x, y, rtol=rtol, atol=atol)
        
        return are_close, max_diff
    
    return _compare 
