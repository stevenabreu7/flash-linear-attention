import torch
import torch.nn as nn
from typing import List, Dict, Optional, Union, Callable
from transformers.utils import logging
from fla.modules.prunable_linear import PrunableLinear

logger = logging.get_logger(__name__)


class IterativeMagnitudePruner:
    def __init__(
        self,
        model: nn.Module,
        target_sparsity: float = 0.5,
        pruning_start_step: int = 1000,
        pruning_end_step: int = 10000,
        pruning_frequency: int = 1000,
        module_name_filter: Optional[Callable[[str], bool]] = None,
        polynomial_degree: float = 3,
    ):
        """
        Iterative Magnitude Pruning scheduler
        
        Args:
            model: The model to prune
            target_sparsity: Final sparsity target (0.0-1.0)
            pruning_start_step: Training step to start pruning
            pruning_end_step: Training step to end pruning ramp-up
            pruning_frequency: How often to update pruning masks (in steps)
            module_name_filter: Function to filter which modules to prune
            polynomial_degree: Degree of the polynomial for sparsity ramp-up
        """
        self.model = model
        self.target_sparsity = target_sparsity
        self.pruning_start_step = pruning_start_step
        self.pruning_end_step = pruning_end_step
        self.pruning_frequency = pruning_frequency
        self.polynomial_degree = polynomial_degree
        
        # Default filter: prune modules with 'mlp' or 'linear' in their name
        self.module_name_filter = module_name_filter or (lambda x: 'mlp' in x.lower() or 'linear' in x.lower())
        
        self.prunable_modules = self._find_prunable_modules()
        self.current_step = 0
        self.current_sparsity = 0.0
        
        logger.info(f"Found {len(self.prunable_modules)} prunable modules")
        
    def _find_prunable_modules(self):
        prunable_modules = []
        for name, module in self.model.named_modules():
            if hasattr(module, 'enable_pruning') and self.module_name_filter(name):
                prunable_modules.append((name, module))
        return prunable_modules
    
    def step(self, training_step: int, log: bool = False):
        """Update pruning masks based on current training step"""
        self.current_step = training_step
        
        # Check if we should update pruning masks
        if (training_step < self.pruning_start_step or 
            training_step > self.pruning_end_step or
            (training_step - self.pruning_start_step) % self.pruning_frequency != 0):
            logger.debug(f"Skipping pruning step {training_step} - not in pruning range")
            return
        
        logger.debug(f"Pruning step {training_step} is in pruning range")
        
        # Calculate target sparsity for this step
        progress = min(1.0, (training_step - self.pruning_start_step) / 
                      (self.pruning_end_step - self.pruning_start_step))
        step_sparsity = self.target_sparsity * (1.0 - (1.0 - progress) ** self.polynomial_degree)

        if step_sparsity <= self.current_sparsity:
            logger.debug(f"Skipping pruning step {training_step} - target sparsity {step_sparsity:.4f} already reached: {self.current_sparsity:.4f}")
            return

        self.current_sparsity = step_sparsity
        logger.info(f"Updating pruning masks to target sparsity: {step_sparsity:.4f}")
        
        # Enable pruning on all modules
        for _, module in self.prunable_modules:
            module.enable_pruning(True)
        
        # Update masks for each prunable module
        for name, module in self.prunable_modules:
            self._update_module_masks(module, step_sparsity)

        if log:
            # Log pruning statistics
            self._log_pruning_stats()
    
    def _update_module_masks(self, module, sparsity):
        """Update masks for all prunable linear layers in a module"""
        for name, child in module.named_modules():
            if isinstance(child, PrunableLinear):
                # Ensure mask has correct dtype
                child.update_mask_dtype()
                
                # Check tensor types and handle accordingly
                weight = child.weight
                mask = child.mask
                weight_data = weight._local_tensor if hasattr(weight, '_local_tensor') else weight
                mask_data = mask._local_tensor if hasattr(mask, '_local_tensor') else mask
                
                # Update the mask based on weight magnitudes
                self._update_mask(weight_data, mask_data, sparsity)
                
                # Enable pruning on this layer
                child.enable_pruning(True)
    
    def _update_mask(self, weight, mask, sparsity):
        """Update a single mask based on weight magnitudes"""
        # Get weight magnitudes
        with torch.no_grad():
            magnitude = weight.abs()
            
            # Calculate threshold for pruning
            k = int(weight.numel() * sparsity)
            if k == 0:
                return
                
            # Find threshold value
            threshold = torch.kthvalue(magnitude.view(-1), k).values
            
            # Update mask (keep weights with magnitude > threshold)
            new_mask = (magnitude > threshold).to(dtype=mask.dtype, device=mask.device)
            mask.data.copy_(new_mask)
    
    def _log_pruning_stats(self, detailed: bool = False):
        """Log pruning statistics"""
        sparsity, nparams, nzparams, full_state = self.get_pruning_stats()
        logger.info(f"Pruning stats: {sparsity=:.4f} ({nzparams:,}/{nparams:,} params pruned)")
        if detailed:
            for name, state in full_state.items():
                logger.info(f"Module {name}: {state}")

    def get_pruning_stats(self, return_full_state: bool = False):
        """Calculate pruning statistics"""
        total_params = 0
        pruned_params = 0
        full_state = {}
        
        for name, module in self.prunable_modules:
            if hasattr(module, 'get_pruning_state'):
                state = module.get_pruning_state()
                full_state[name] = state
                
                # Count parameters for PrunableLinear
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


def setup_pruning_for_hgrn(model, args):
    """
    Setup pruning for HGRN model based on training arguments
    """
    if not hasattr(model.config, 'use_pruning') or not model.config.use_pruning:
        return None
    
    logger.info(f"Setting up pruning with target sparsity {model.config.target_sparsity}")
    
    for name, module in model.named_modules():
        if hasattr(module, 'enable_pruning'):
            module.enable_pruning(True)
            logger.info(f"Enabled pruning for module {name}")
    
    return model.pruner


def update_pruning(model, step):
    """Update pruning masks for a model that supports pruning"""
    if hasattr(model, 'pruner') and model.pruner is not None:
        model.pruner.step(step)
    elif hasattr(model, 'update_pruning'):
        model.update_pruning(step)
