import torch
import torch.nn as nn
import pytest
from fla.modules.prunable_linear import PrunableLinear
from fla.rigl_pruning import RigLPruner


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = PrunableLinear(10, 20)
        self.fc2 = PrunableLinear(20, 10)
        
    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x


def test_rigl_pruner_initialization():
    model = SimpleModel()
    pruner = RigLPruner(
        model=model, 
        target_sparsity=0.5,
        pruning_start_step=10,
        pruning_end_step=100,
        pruning_frequency=10
    )
    
    # Check initialization
    assert len(pruner.prunable_modules) == 2
    assert pruner.target_sparsity == 0.5
    assert pruner.current_sparsity == 0.0
    assert pruner.current_step == 0
    
    # Check that all modules are enabled for pruning
    for name, module in pruner.prunable_modules:
        assert module.pruning_active
        assert torch.all(module.mask == 1.0)  # All weights are initially kept


def test_rigl_update_connectivity():
    torch.manual_seed(42)  # For reproducibility
    
    model = SimpleModel()
    pruner = RigLPruner(
        model=model, 
        target_sparsity=0.5,
        pruning_start_step=10,
        pruning_end_step=100,
        pruning_frequency=10
    )
    
    # Prepare optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # Generate some data and do a forward/backward pass to populate gradients
    inputs = torch.randn(4, 10)
    outputs = model(inputs)
    loss = outputs.mean()
    loss.backward()
    
    # Accumulate gradients
    pruner.accumulate_gradients()
    
    # Check mask before update
    initial_mask_fc1 = model.fc1.mask.clone()
    initial_mask_fc2 = model.fc2.mask.clone()
    assert torch.all(initial_mask_fc1 == 1.0)
    assert torch.all(initial_mask_fc2 == 1.0)
    
    # Perform RIGL step at the right step
    pruner.current_step = 10  # Set to pruning_start_step
    should_optimize = pruner.step(10, optimizer)
    
    # Check that we should skip optimizer step
    assert not should_optimize
    
    # Check masks after update - should have 50% sparsity
    assert (model.fc1.mask == 0).float().mean().item() > 0.45  # Allow some rounding tolerance
    assert (model.fc1.mask == 0).float().mean().item() < 0.55
    assert (model.fc2.mask == 0).float().mean().item() > 0.45
    assert (model.fc2.mask == 0).float().mean().item() < 0.55
    
    # Check the mask is actually different
    assert not torch.all(model.fc1.mask == initial_mask_fc1)
    assert not torch.all(model.fc2.mask == initial_mask_fc2)


def test_rigl_gradient_accumulation():
    torch.manual_seed(42)  # For reproducibility
    
    model = SimpleModel()
    pruner = RigLPruner(
        model=model, 
        target_sparsity=0.5,
        pruning_start_step=30,
        pruning_end_step=100,
        pruning_frequency=10,
        gradient_accumulation_n=5
    )
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # Check that accumulation doesn't happen far from update step
    pruner.current_step = 10
    inputs = torch.randn(4, 10)
    outputs = model(inputs)
    loss = outputs.mean()
    loss.backward()
    pruner.accumulate_gradients()
    
    for name in pruner.grad_accum:
        assert pruner.grad_accum[name] is None
    
    # Check that accumulation happens near update step
    pruner.current_step = 26  # 4 steps away from 30
    inputs = torch.randn(4, 10)
    outputs = model(inputs)
    loss = outputs.mean()
    loss.backward()
    pruner.accumulate_gradients()
    
    # Accumulation should happen since we're within gradient_accumulation_n
    assert any(pruner.grad_accum[name] is not None for name in pruner.grad_accum)


def test_rigl_full_cycle():
    torch.manual_seed(42)  # For reproducibility
    
    model = SimpleModel()
    pruner = RigLPruner(
        model=model, 
        target_sparsity=0.8,
        pruning_start_step=10,
        pruning_end_step=100,
        pruning_frequency=10
    )
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # Initial sparsity should be 0
    assert pruner.current_sparsity == 0.0
    
    # Run a full training cycle
    for step in range(1, 120):
        pruner.current_step = step
        
        # Generate some data and do forward/backward pass
        inputs = torch.randn(4, 10)
        outputs = model(inputs)
        loss = outputs.mean()
        loss.backward()
        
        # Accumulate gradients
        pruner.accumulate_gradients()
        
        # Check if this is a RigL step
        should_optimize = pruner.step(step, optimizer)
        
        # Check sparsity at key steps
        if step == 10:  # Starting point
            # Should have applied initial sparsity
            overall_sparsity, _, _, _ = pruner.get_pruning_stats()
            assert 0.75 < overall_sparsity < 0.85  # Around 80%, allow rounding
        
        if step == 50:  # Middle of pruning
            # Still at target sparsity
            overall_sparsity, _, _, _ = pruner.get_pruning_stats()
            assert 0.75 < overall_sparsity < 0.85
        
        if step == 100:  # End of pruning
            # Still at target sparsity
            overall_sparsity, _, _, _ = pruner.get_pruning_stats()
            assert 0.75 < overall_sparsity < 0.85
        
        if step > 100:  # After pruning
            # Still at target sparsity but no more updates
            overall_sparsity, _, _, _ = pruner.get_pruning_stats()
            assert 0.75 < overall_sparsity < 0.85
            assert should_optimize  # Should allow optimizer step
        
        if should_optimize:
            optimizer.step()
            optimizer.zero_grad()


def test_rigl_different_alpha():
    torch.manual_seed(42)  # For reproducibility
    
    model = SimpleModel()
    pruner = RigLPruner(
        model=model, 
        target_sparsity=0.8,
        pruning_start_step=10,
        pruning_end_step=100,
        pruning_frequency=10,
        alpha=0.1  # More conservative
    )
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # Store initial state
    fc1_initial_mask = model.fc1.mask.clone()
    fc2_initial_mask = model.fc2.mask.clone()
    
    # Run a few steps
    for step in range(10, 31, 10):
        inputs = torch.randn(4, 10)
        outputs = model(inputs)
        loss = outputs.mean()
        loss.backward()
        pruner.accumulate_gradients()
        pruner.step(step, optimizer)
    
    # Store new masks
    fc1_new_mask = model.fc1.mask.clone()
    fc2_new_mask = model.fc2.mask.clone()
    
    # Compare with a higher alpha
    torch.manual_seed(42)  # Reset for fair comparison
    model = SimpleModel()
    pruner_higher_alpha = RigLPruner(
        model=model, 
        target_sparsity=0.8,
        pruning_start_step=10,
        pruning_end_step=100,
        pruning_frequency=10,
        alpha=0.5  # More aggressive
    )
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # Run the same steps
    for step in range(10, 31, 10):
        inputs = torch.randn(4, 10)
        outputs = model(inputs)
        loss = outputs.mean()
        loss.backward()
        pruner_higher_alpha.accumulate_gradients()
        pruner_higher_alpha.step(step, optimizer)
    
    # Calculate mask changes
    changes_low_alpha = (fc1_initial_mask != fc1_new_mask).float().mean().item() + \
                        (fc2_initial_mask != fc2_new_mask).float().mean().item()
    
    changes_high_alpha = (fc1_initial_mask != model.fc1.mask).float().mean().item() + \
                         (fc2_initial_mask != model.fc2.mask).float().mean().item()
    
    # Higher alpha should cause more changes to the mask
    assert changes_high_alpha > changes_low_alpha


if __name__ == "__main__":
    test_rigl_pruner_initialization()
    test_rigl_update_connectivity()
    test_rigl_gradient_accumulation()
    test_rigl_full_cycle()
    test_rigl_different_alpha()
    print("All tests passed!")