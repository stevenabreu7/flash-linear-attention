import torch
import torch.nn as nn
from typing import List, Dict, Optional, Union, Callable
from transformers.utils import logging
from fla.modules.prunable_linear import PrunableLinear
import numpy as np

logger = logging.get_logger(__name__)


class RigLPruner:
    """
    RigL (Rigging the Lottery) sparse training scheduler for dynamic network topology updates.
    
    Based on the paper: "Rigging the Lottery: Making All Tickets Winners" by Evci et al.
    https://arxiv.org/abs/1911.11134
    
    This implementation adapts the RigL algorithm to work with the Flame codebase,
    integrating with existing PrunableLinear layers and training infrastructure.
    
    RigL dynamically updates network connectivity during training by pruning weights
    with small magnitudes and growing new connections based on gradient information.
    """
    def __init__(
        self,
        model: nn.Module,
        target_sparsity: float = 0.5,
        pruning_start_step: int = 1000,
        pruning_end_step: int = 10000,
        pruning_frequency: int = 100,
        module_name_filter: Optional[Callable[[str], bool]] = None,
        gradient_accumulation_n: int = 1,
        alpha: float = 0.3,
        ignore_linear_layers: bool = False
    ):
        """
        Args:
            model: The model to prune
            target_sparsity: Final sparsity target (0.0-1.0)
            pruning_start_step: Training step to start pruning
            pruning_end_step: Training step to end pruning/topology updates
            pruning_frequency: How often to update network topology (in steps)
            module_name_filter: Function to filter which modules to prune
            gradient_accumulation_n: Number of batches to accumulate gradients from
            alpha: Fraction of connections to rewire at each update (lower = more conservative)
            ignore_linear_layers: If True, don't prune final classification/projection layers
        """
        if target_sparsity < 0.0 or target_sparsity >= 1.0:
            raise ValueError(f"Target sparsity must be in range [0.0, 1.0), got {target_sparsity}")
        
        self.model = model
        self.target_sparsity = target_sparsity
        self.pruning_start_step = pruning_start_step
        self.pruning_end_step = pruning_end_step
        self.pruning_frequency = pruning_frequency
        self.gradient_accumulation_n = gradient_accumulation_n
        self.alpha = alpha
        self.ignore_linear_layers = ignore_linear_layers
        
        # Default filter: prune modules with 'mlp' or 'linear' in their name
        self.module_name_filter = module_name_filter or (lambda x: 'mlp' in x.lower() or 'linear' in x.lower())
        
        # Find all modules that can be pruned
        self.prunable_modules = self._find_prunable_modules()
        
        # Track state
        self.current_step = 0
        self.rigl_steps = 0
        self.current_sparsity = 0.0
        
        # Initialize gradient accumulation
        self.grad_accum = {}
        self.grad_accum_count = {}
        for name, module in self.prunable_modules:
            if isinstance(module, PrunableLinear):
                self.grad_accum[name] = None
                self.grad_accum_count[name] = 0
        
        logger.info(f"Found {len(self.prunable_modules)} prunable modules for RigL")
            
    def _find_prunable_modules(self):
        """Find all modules that can be pruned with RigL"""
        prunable_modules = []
        for name, module in self.model.named_modules():
            if hasattr(module, 'enable_pruning') and self.module_name_filter(name):
                # Skip final linear layers if specified
                if self.ignore_linear_layers and 'linear' in name.lower() and ('output' in name.lower() or 'head' in name.lower()):
                    continue
                prunable_modules.append((name, module))
        return prunable_modules
    
    def step(self, training_step: int, optimizer=None):
        """Update pruning masks based on current training step and gradients
        
        Args:
            training_step: Current training step
            optimizer: Optimizer (needed to access momentum buffers)
            
        Returns:
            bool: True if a normal optimizer step should be performed, False if RigL
                  performed a topology update (no optimizer step needed)
        """
        self.current_step = training_step
        
        # Check if we are in the pruning phase
        if training_step < self.pruning_start_step or training_step > self.pruning_end_step:
            return True
        
        # Check if this is a RigL update step
        if (training_step - self.pruning_start_step) % self.pruning_frequency != 0:
            return True
        
        # Enable pruning on all modules
        for _, module in self.prunable_modules:
            module.enable_pruning(True)
        
        # Update sparsity based on pruning schedule
        progress = min(1.0, (training_step - self.pruning_start_step) / 
                      (self.pruning_end_step - self.pruning_start_step))
        self.current_sparsity = self.target_sparsity
        
        # Calculate drop fraction (cosine annealing)
        drop_fraction = self.alpha / 2 * (1 + np.cos((progress * np.pi)))

        logger.info(f"Performing RigL update at step {training_step}, " 
                    f"current sparsity: {self.current_sparsity:.4f}, "
                    f"drop fraction: {drop_fraction:.4f}")
        
        # Update connectivity for each module
        self._update_connectivity(drop_fraction, optimizer)
        
        # Track RigL steps
        self.rigl_steps += 1
        
        # Clear gradient accumulation
        for name in self.grad_accum:
            self.grad_accum[name] = None
            self.grad_accum_count[name] = 0
            
        # Return False to indicate no optimizer step should be performed
        return False
    
    def accumulate_gradients(self):
        """Accumulate gradients for potential regrowth
        
        This should be called before the optimizer step, after backward pass.
        """
        # Check if we should accumulate gradients 
        # (only if approaching a RigL step within grad_accumulation_n steps)
        if self.current_step < self.pruning_start_step or self.current_step > self.pruning_end_step:
            return
            
        next_rigl_step = self.pruning_start_step + ((self.current_step - self.pruning_start_step) 
                                                   // self.pruning_frequency + 1) * self.pruning_frequency
        steps_to_next = next_rigl_step - self.current_step
        
        if steps_to_next > self.gradient_accumulation_n:
            return
        
        # Accumulate gradients from current batch
        for name, module in self.prunable_modules:
            if not isinstance(module, PrunableLinear):
                continue
                
            if hasattr(module.weight, 'grad') and module.weight.grad is not None:
                grad = module.weight.grad.abs().clone()
                
                if self.grad_accum[name] is None:
                    self.grad_accum[name] = grad
                else:
                    self.grad_accum[name] += grad
                    
                self.grad_accum_count[name] += 1
    
    def _update_connectivity(self, drop_fraction, optimizer):
        """Update network connectivity using RigL algorithm
        
        Args:
            drop_fraction: Fraction of active connections to drop
            optimizer: Optimizer for resetting momentum buffers
        """
        with torch.no_grad():
            for name, module in self.prunable_modules:
                if not isinstance(module, PrunableLinear):
                    continue
                
                # Skip modules with no sparsity
                if self.target_sparsity <= 0.0:
                    continue
                
                # Get weight and mask tensors
                weight = module.weight
                current_mask = module.mask
                
                # Calculate scores for dropping connections (weight magnitude)
                score_drop = torch.abs(weight)
                
                # Calculate scores for growing connections (gradient information)
                if name in self.grad_accum and self.grad_accum[name] is not None:
                    # Use accumulated gradients if available
                    count = max(1, self.grad_accum_count[name])
                    score_grow = self.grad_accum[name] / count
                elif hasattr(weight, 'grad') and weight.grad is not None:
                    # Otherwise use current gradients
                    score_grow = torch.abs(weight.grad)
                else:
                    # If no gradients available, skip this module
                    logger.warning(f"No gradients available for module {name}, skipping RigL update")
                    continue
                
                # Calculate drop/grow quantities
                n_total = weight.numel()
                n_nonzero = torch.sum(current_mask).item()
                n_prune = int(n_nonzero * drop_fraction)
                
                if n_prune == 0:
                    continue
                
                # Create drop mask (keep largest weights)
                weight_flat = score_drop.view(-1)
                mask_flat = current_mask.view(-1)
                
                # Only consider active connections for pruning
                active_indices = torch.nonzero(mask_flat, as_tuple=True)[0]
                if len(active_indices) == 0:
                    continue
                    
                active_weights = weight_flat[active_indices]
                _, sorted_indices = torch.sort(active_weights)
                drop_indices = active_indices[sorted_indices[:n_prune]]
                
                # Create grow mask based on gradient info
                score_grow_flat = score_grow.view(-1)
                
                # Only consider inactive connections for growth
                inactive_indices = torch.nonzero(mask_flat == 0, as_tuple=True)[0]
                if len(inactive_indices) < n_prune:
                    # Not enough inactive connections to maintain parameter count
                    n_prune = len(inactive_indices)
                    
                if n_prune == 0 or len(inactive_indices) == 0:
                    continue
                    
                inactive_grads = score_grow_flat[inactive_indices]
                _, sorted_indices = torch.sort(inactive_grads, descending=True)
                grow_indices = inactive_indices[sorted_indices[:n_prune]]
                
                # Update mask
                new_mask = mask_flat.clone()
                new_mask[drop_indices] = 0  # Drop connections
                new_mask[grow_indices] = 1  # Grow connections
                module.mask.data = new_mask.view_as(current_mask)
                
                # Reset momentum buffers for the regrown connections
                if optimizer is not None:
                    param_state = optimizer.state.get(weight, None)
                    if param_state is not None and 'momentum_buffer' in param_state:
                        momentum = param_state['momentum_buffer']
                        momentum_flat = momentum.view(-1)
                        momentum_flat[grow_indices] = 0
                
                # Apply mask to weights
                module.weight.data = module.weight.data * module.mask.data
    
    def get_pruning_stats(self):
        """Calculate pruning statistics
        
        Returns:
            tuple: (overall_sparsity, nparams, nzparams, full_state)
                overall_sparsity: Current overall sparsity
                nparams: Total number of parameters
                nzparams: Number of pruned parameters
                full_state: Detailed state for each module
        """
        total_params = 0
        pruned_params = 0
        full_state = {}
        
        for name, module in self.prunable_modules:
            if hasattr(module, 'get_pruning_state'):
                state = module.get_pruning_state()
                full_state[name] = state
                
                # Count parameters
                if hasattr(module, 'mask'):
                    mask_numel = module.mask.numel()
                    # Handle DTensor case
                    if hasattr(module.mask, '_local_tensor'):
                        # Get local counts
                        local_mask = module.mask._local_tensor
                        local_zeros = (local_mask == 0).sum().item()
                        local_total = local_mask.numel()
                        
                        # Gather counts from all processes
                        zeros_tensor = torch.tensor([local_zeros], dtype=torch.float64, device=local_mask.device)
                        total_tensor = torch.tensor([local_total], dtype=torch.float64, device=local_mask.device)
                        
                        torch.distributed.all_reduce(zeros_tensor, op=torch.distributed.ReduceOp.SUM)
                        torch.distributed.all_reduce(total_tensor, op=torch.distributed.ReduceOp.SUM)
                        
                        pruned_params += zeros_tensor.item() if module.pruning_active else 0
                        total_params += total_tensor.item()
                    else:
                        total_params += mask_numel
                        pruned_params += (module.mask == 0).sum().item() if module.pruning_active else 0
        
        overall_sparsity = 0.0 if total_params <= 0 else pruned_params / total_params
        return overall_sparsity, total_params, pruned_params, full_state
    
    def __str__(self):
        """String representation with statistics"""
        s = 'RigLPruner(\n'
        s += f'  target_sparsity={self.target_sparsity:.4f},\n'
        s += f'  current_sparsity={self.current_sparsity:.4f},\n'
        s += f'  step={self.current_step},\n'
        s += f'  rigl_steps={self.rigl_steps},\n'
        s += f'  pruning_start_step={self.pruning_start_step},\n'
        s += f'  pruning_end_step={self.pruning_end_step},\n'
        s += f'  pruning_frequency={self.pruning_frequency},\n'
        s += f'  alpha={self.alpha},\n'
        s += f'  prunable_modules={len(self.prunable_modules)}\n'
        s += ')'
        return s