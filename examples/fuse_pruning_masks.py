#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example script demonstrating how to fuse and unfuse pruning masks in a model.
This converts between sparse models (with explicit masking) and normal models 
(with zeros in the pruned weight positions).
"""

import argparse
import os
import logging
import torch
from transformers import AutoTokenizer

from fla.models.hgrn.modeling_hgrn import HGRNForCausalLM
from fla.models.hgrn.configuration_hgrn import HGRNConfig
from fla.pruning import fuse_pruning_masks, unfuse_pruning_masks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Fuse or unfuse pruning masks in a model")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the converted model",
    )
    parser.add_argument(
        "--action",
        type=str,
        choices=["fuse", "unfuse"],
        default="fuse",
        help="Whether to fuse or unfuse pruning masks",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    logger.info(f"Loading model from {args.model_path}")
    
    # Load model
    # In a real scenario, you might want to detect the model type
    model = HGRNForCausalLM.from_pretrained(args.model_path)
    
    # Check if the model has pruning enabled
    if not hasattr(model.config, 'use_pruning') or not model.config.use_pruning:
        logger.warning("Model does not have pruning enabled in its config")
        
    # Get sparsity info before the operation
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model has {total_params:,} parameters")
    
    # Perform the requested action
    if args.action == "fuse":
        logger.info("Fusing pruning masks into weights...")
        fused_count = fuse_pruning_masks(model)
        logger.info(f"Fused {fused_count} prunable layers")
    else:
        logger.info("Unfusing pruning masks from weights...")
        unfused_count = unfuse_pruning_masks(model)
        logger.info(f"Unfused {unfused_count} prunable layers")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save the model
    logger.info(f"Saving model to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    
    # Try to save tokenizer if available
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        tokenizer.save_pretrained(args.output_dir)
        logger.info("Tokenizer saved successfully")
    except Exception as e:
        logger.warning(f"Failed to save tokenizer: {e}")
    
    logger.info("Done!")


if __name__ == "__main__":
    main() 
