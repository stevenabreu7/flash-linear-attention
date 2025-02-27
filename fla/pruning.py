import torch
import torch.nn as nn
from typing import List, Dict, Optional, Union, Callable
from transformers.utils import logging

logger = logging.get_logger(__name__)


class IterativeMagnitudePruner:
    def __init__(
        self,
        model: nn.Module,
        target_sparsity: float = 0.5,
        pruning_steps: int = 10,
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
            pruning_steps: Number of pruning steps to reach target sparsity
            pruning_start_step: Training step to start pruning
            pruning_end_step: Training step to end pruning ramp-up
            pruning_frequency: How often to update pruning masks (in steps)
            module_name_filter: Function to filter which modules to prune
            polynomial_degree: Degree of the polynomial for sparsity ramp-up
        """
        self.model = model
        self.target_sparsity = target_sparsity
        self.pruning_steps = pruning_steps
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
            return
        
        # Calculate target sparsity for this step
        progress = min(1.0, (training_step - self.pruning_start_step) / 
                      (self.pruning_end_step - self.pruning_start_step))
        step_sparsity = self.target_sparsity * (1.0 - (1.0 - progress) ** self.polynomial_degree)

        if step_sparsity <= self.current_sparsity:
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
        """Update pruning masks for a single module based on weight magnitudes"""
        # Handle PrunableLinear
        if hasattr(module, 'mask') and hasattr(module, 'weight'):
            self._update_mask(module.weight, module.mask, sparsity)
        
        # Handle PrunableGatedMLP
        if hasattr(module, 'gate_mask'):
            self._update_mask(module.gate_proj.weight, module.gate_mask, sparsity)
            
        if hasattr(module, 'up_mask'):
            self._update_mask(module.up_proj.weight, module.up_mask, sparsity)
            
        if hasattr(module, 'down_mask'):
            self._update_mask(module.down_proj.weight, module.down_mask, sparsity)
    
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
            new_mask = (magnitude > threshold).float()
            mask.copy_(new_mask)
    
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
                    total_params += module.mask.numel()
                    pruned_params += (module.mask == 0).sum().item()
                
                # Count parameters for PrunableGatedMLP
                if hasattr(module, 'gate_mask'):
                    total_params += module.gate_mask.numel()
                    pruned_params += (module.gate_mask == 0).sum().item()
                if hasattr(module, 'up_mask'):
                    total_params += module.up_mask.numel()
                    pruned_params += (module.up_mask == 0).sum().item()
                if hasattr(module, 'down_mask'):
                    total_params += module.down_mask.numel()
                    pruned_params += (module.down_mask == 0).sum().item()
        
        overall_sparsity = None if total_params <= 0 else pruned_params / total_params
        return overall_sparsity, total_params, pruned_params, full_state


# Add helper functions for saving and loading pruned models
def save_pruned_model(model, path):
    """Save model with pruning masks"""
    # Save model state dict
    state_dict = model.state_dict()
    torch.save(state_dict, path)


def load_pruned_model(model, path):
    """Load model with pruning masks and activate pruning"""
    # Load state dict
    state_dict = torch.load(path)
    model.load_state_dict(state_dict)
    
    # Enable pruning on all prunable modules based on config
    if hasattr(model.config, 'use_pruning') and model.config.use_pruning:
        for name, module in model.named_modules():
            if hasattr(module, 'enable_pruning'):
                module.enable_pruning(True)
                
        logger.info("Enabled pruning on loaded model") 


def setup_pruning_for_hgrn(model, args):
    """
    Setup pruning for HGRN model based on training arguments
    
    Args:
        model: The HGRN model
        args: Training arguments with pruning configuration
    """
    if not hasattr(model.config, 'use_pruning') or not model.config.use_pruning:
        return None
    
    # Pruning is configured in the model config
    logger.info(f"Setting up pruning with target sparsity {model.config.target_sparsity}")
    
    # Enable pruning on all prunable modules
    for name, module in model.named_modules():
        if hasattr(module, 'enable_pruning'):
            module.enable_pruning(True)
            logger.info(f"Enabled pruning for module {name}")
    
    return model.pruner


def update_pruning(model, step):
    """
    Update pruning masks during training
    
    Args:
        model: The model with pruning
        step: Current training step
    """
    if hasattr(model, 'update_pruning'):
        model.update_pruning(step) 
