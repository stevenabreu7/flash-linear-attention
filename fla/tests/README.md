# FLA Component Tests

This directory contains tests for the components used in the Flash-Linear-Attention (FLA) library, particularly for HGRN (Hierarchically Gated Recurrent Neural Network) model components.

## Overview

These tests are designed to verify that the optimized components in FLA match the behavior of simple reference implementations. The tests focus on:

1. **HGRNAttention** - The hierarchical gated recurrent attention mechanism
2. **GatedMLP** - The SwiGLU-based MLP layer
3. **PrunableLinear** - The prunable linear layer used for sparse training
4. **RMSNorm** - The root mean square normalization layer

Each test compares a simple PyTorch-only implementation against the optimized FLA implementation to ensure functional equivalence.

## Running the Tests

You can run the tests using the provided `run_tests.py` script:

```bash
# Run all tests
python fla/tests/run_tests.py

# Run with verbose output
python fla/tests/run_tests.py -v

# Run only float16 tests
python fla/tests/run_tests.py --precision float16

# Run only bfloat16 tests
python fla/tests/run_tests.py --precision bfloat16

# Run specific test file
python fla/tests/run_tests.py --test-file test_hgrn_components.py
```

Or you can use pytest directly:

```bash
# Run all tests
pytest fla/tests/

# Run specific test file
pytest fla/tests/test_hgrn_components.py

# Run specific test
pytest fla/tests/test_hgrn_components.py::test_rmsnorm
```

## Test Details

### test_hgrn_components.py

This file contains tests for all the key HGRN components:

- `test_rmsnorm`: Tests that the FLA RMSNorm produces the same results as a reference implementation
- `test_gated_mlp`: Tests that the GatedMLP with SwiGLU activation matches a reference implementation
- `test_prunable_linear`: Tests that the PrunableLinear layer correctly applies pruning masks
- `test_hgrn_attention`: Tests that HGRNAttention produces results consistent with a reference implementation

Each test is run with different combinations of parameters (batch sizes, sequence lengths, dimensions) and with both float16 and bfloat16 precision to ensure robustness.

## Note on Precision

These tests specifically focus on mixed-precision training compatibility, testing with both float16 and bfloat16. This ensures the components will work correctly during sparse HGRN training with reduced precision.

For the HGRNAttention test, note that the reference implementation is a simplified version that captures the mathematical behavior but not the exact optimized computation. Therefore, the test mainly checks for shape consistency and general value ranges rather than exact numerical equivalence. 
