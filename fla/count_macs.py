#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

def count_macs_elementwise(size):
    """Count MACs in element-wise multiplications (each element is 1 MAC)"""
    return size

def count_macs_linear(in_features, out_features, sparsity=0.0):
    """Count MACs in a linear layer, accounting for sparsity"""
    # Each output element needs in_features MACs
    # With sparsity, we have fewer effective operations
    total_params = in_features * out_features
    active_params = total_params * (1 - sparsity)
    return active_params

def calculate_macs_hgrn(config, sparsity=0.0):
    """Calculate MACs for a single token inference with HGRN model"""
    total_macs = 0
    
    # Model dimensions
    h = config["hidden_size"]
    hidden_ratio = config.get("hidden_ratio", 4)
    i = config.get("intermediate_size") or int(h * hidden_ratio * 2 / 3)
    i = 256 * ((i + 256 - 1) // 256)  # Align to 256 as in GatedMLP
    n_layers = config["num_hidden_layers"]
    expand_ratio = config.get("expand_ratio", 1)
    attn = config.get("attn", None)
    vocab_size = config.get("vocab_size", 32000)
    
    # For each layer
    for layer_idx in range(n_layers):
        layer_macs = 0
        
        # 1. Attention block
        if attn is not None and layer_idx in attn.get('layers', []):
            # Traditional attention (note: we're not calculating full attention ops here, 
            # just the matrix multiplications)
            n_heads = attn['num_heads']
            n_kv_heads = attn.get('num_kv_heads', n_heads)
            head_dim = h // n_heads
            
            # QKV projections
            layer_macs += count_macs_linear(h, h * 2 + n_kv_heads * head_dim * 2, sparsity)
            # Output projection
            layer_macs += count_macs_linear(h, h, sparsity)
        else:
            # HGRN Attention
            input_dim = int(h * expand_ratio)
            
            # i_proj, f_proj, g_proj
            layer_macs += count_macs_linear(h, input_dim, sparsity) * 3
            
            # HGRN recurrent operations (element-wise multiplications)
            # Element-wise ops in HGRN attention
            layer_macs += count_macs_elementwise(input_dim)
            
            # o_proj
            layer_macs += count_macs_linear(input_dim, h, sparsity)
        
        # 2. MLP block
        # gate_proj, up_proj
        layer_macs += count_macs_linear(h, i, sparsity) * 2
        # Element-wise multiply in SwiGLU
        layer_macs += count_macs_elementwise(i)
        # down_proj
        layer_macs += count_macs_linear(i, h, sparsity)
        
        # Add this layer's MACs to total
        total_macs += layer_macs
    
    # Final output projection (lm_head)
    # NOTE: embedding matrices are not pruned
    total_macs += count_macs_linear(h, vocab_size, 0.0)
    
    return total_macs

def main():
    parser = argparse.ArgumentParser(description='Calculate MACs for HGRN model')
    parser.add_argument('--sparsity', type=float, default=0.0, help='Manual sparsity value (0.0-1.0)')
    
    # Example config parameters if no config file provided
    parser.add_argument('--hidden_size', type=int, default=2048)
    parser.add_argument('--hidden_ratio', type=int, default=4)
    parser.add_argument('--num_hidden_layers', type=int, default=24)
    parser.add_argument('--vocab_size', type=int, default=32000)
    parser.add_argument('--expand_ratio', type=int, default=1)
    parser.add_argument('--use_attention', action='store_true', help='Use standard attention instead of HGRN')
    parser.add_argument('--num_heads', type=int, default=16, help='Number of attention heads')
    parser.add_argument('--num_kv_heads', type=int, default=None, help='Number of KV heads for grouped query attention')
    
    args = parser.parse_args()
    
    # Create config dictionary
    config = {
        "hidden_size": args.hidden_size,
        "hidden_ratio": args.hidden_ratio,
        "num_hidden_layers": args.num_hidden_layers,
        "vocab_size": args.vocab_size,
        "expand_ratio": args.expand_ratio,
    }
    
    # Add attention config if needed
    if args.use_attention:
        num_kv_heads = args.num_kv_heads or args.num_heads
        config["attn"] = {
            "layers": list(range(args.num_hidden_layers)),  # All layers use attention
            "num_heads": args.num_heads,
            "num_kv_heads": num_kv_heads
        }
    
    # Calculate MACs
    macs = calculate_macs_hgrn(config, args.sparsity)
    
    # Print results
    print("\nHGRN Model MAC Calculation Results:")
    print(f"Hidden Size: {config['hidden_size']}")
    print(f"Number of Layers: {config['num_hidden_layers']}")
    print(f"Expand Ratio: {config['expand_ratio']}")
    print(f"Sparsity: {args.sparsity:.4f}")
    
    if args.use_attention:
        print(f"Using standard attention with {args.num_heads} heads")
        if args.num_kv_heads and args.num_kv_heads != args.num_heads:
            print(f"Using grouped-query attention with {args.num_kv_heads} KV heads")
    
    # Format large numbers with commas
    print(f"\nTotal MACs per token: {macs:,}")
    print(f"Total GMACs per token: {macs/1e9:.4f}")
    
    # Estimate FLOPS (2 FLOPS per MAC)
    flops = macs * 2
    print(f"Total FLOPs per token: {flops:,}")
    print(f"Total GFLOPs per token: {flops/1e9:.4f}")

if __name__ == "__main__":
    main() 
