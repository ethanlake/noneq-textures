#!/usr/bin/env python3
"""
Convert PyTorch .pt model files to JSON format for NoiseNCA demo.

Usage:
    python convert_pt_to_json.py <input.pt> [output.json] [--noise-level FLOAT] [--pos-emb]

This script follows the conversion convention from export.ipynb.
It expects PyTorch state_dict with keys like 'w1.weight', 'w1.bias', 'w2.weight', etc.
"""

import os
import json
import numpy as np
import torch
import argparse
import sys


def tile2d(a, w=None):
    """
    Tile a 3D array into a 2D grid.
    
    Args:
        a: 3D numpy array of shape [n, h, w, ...]
        w: Width of the grid (if None, uses sqrt of n)
        
    Returns:
        2D tiled array
    """
    a = np.asarray(a)
    if w is None:
        w = int(np.ceil(np.sqrt(len(a))))
    th, tw = a.shape[1:3]
    pad = (w - len(a)) % w
    a = np.pad(a, [(0, pad)] + [(0, 0)] * (a.ndim - 1), 'constant')
    h = len(a) // w
    a = a.reshape([h, w] + list(a.shape[1:]))
    a = np.rollaxis(a, 2, 1).reshape([th * h, tw * w] + list(a.shape[4:]))
    return a


@torch.no_grad()
def torch_model_to_np(state_dict):
    """
    Convert PyTorch state_dict to numpy arrays following the notebook convention.
    
    Returns:
        Tuple of (layers, is_rotation_invariant) where:
        - layers: List of numpy arrays: [layer1_params, layer2_params, ..., noise_level]
        - is_rotation_invariant: Boolean indicating if model uses 2 filters (rotation-invariant)
    """
    layers = []

    # Layer 1: w1.weight and w1.bias
    if 'w1.weight' not in state_dict:
        raise ValueError("Could not find 'w1.weight' in state_dict. "
                        f"Available keys: {list(state_dict.keys())[:10]}...")
    
    w1 = state_dict['w1.weight'][:, :, 0, 0].detach().cpu().numpy()
    # w1 shape: [fc_dim, in_channels] where in_channels is either chn*2 or chn*4
    
    b1 = state_dict['w1.bias'][:, None].detach().cpu().numpy()
    # b1 shape: [fc_dim, 1]
    
    # Layer 2: w2.weight (no bias) - needed to determine chn
    if 'w2.weight' not in state_dict:
        raise ValueError("Could not find 'w2.weight' in state_dict. "
                        f"Available keys: {list(state_dict.keys())[:10]}...")
    
    w2 = state_dict['w2.weight'][:, :, 0, 0].detach().cpu().numpy()
    # w2 shape: [chn, fc_dim] (after extracting [:, :, 0, 0] from [chn, fc_dim, 1, 1])
    
    # Detect rotation-invariant model by checking w1 input channels
    # w1 has shape [fc_dim, in_channels] (after extracting [:, :, 0, 0])
    # w2 has shape [chn, fc_dim] (after extracting [:, :, 0, 0])
    in_channels = w1.shape[1]  # Number of input channels to w1 (excluding bias)
    chn = w2.shape[0]  # Number of channels in the model (output channels from w2)
    
    # Rotation-invariant models have 2 filters: in_channels = chn * 2
    # Standard models have 4 filters: in_channels = chn * 4
    is_rotation_invariant = (in_channels == chn * 2)
    
    # Concatenate and transpose: [chn, fc_dim] -> [chn+1, fc_dim] after adding bias
    layer1_params = np.concatenate([w1, b1], axis=1).T
    # layer1_params shape: [chn+1, fc_dim]
    layer1_params = layer1_params[None, ...]  # Add batch dimension: [1, chn+1, fc_dim]
    layers.append(layer1_params)

    # w2 shape after transpose: [in_channels, out_channels]
    w2 = w2.T
    w2 = w2[None, ...]  # Add batch dimension: [1, in_channels, out_channels]
    layers.append(w2)
    
    # Noise level
    if 'noise_level' in state_dict:
        noise_level = state_dict['noise_level'].item()
    else:
        noise_level = 0.1  # Default
    layers.append(noise_level)

    return layers, is_rotation_invariant


def export_np_models_to_json(np_models, metadata):
    """
    Export numpy models in a form that the JavaScript can read.
    
    Args:
        np_models: List from torch_model_to_np: [layer1, layer2, ..., noise_level]
        metadata: Dict with 'model_names' and 'pos_emb' (bool)
        
    Returns:
        Dictionary in JSON format
    """
    models_js = {
        'model_names': metadata['model_names'],
        'layers': [],
        'noise_level': np_models[-1],
        'rotation_invariant': metadata.get('rotation_invariant', False)
    }
    
    for i, layer in enumerate(np_models[:-1]):
        # layer shape: [n, c_in, fc_dim] where n=1 for single model
        shape = layer[0].shape  # Original shape: [c_in, fc_dim]
        layer = np.array(layer)  # shape: [n, c_in, fc_dim]
        
        if i == 0:
            # First layer: handle pos_emb and rearrange filters
            c = 1  # Bias
            if metadata['pos_emb']:
                c += 2  # Positional embedding channels
            
            # Get rotation-invariant flag from metadata (detected earlier from model structure)
            is_rotation_invariant = metadata.get('rotation_invariant', False)
            filter_channels = layer.shape[1] - c  # Number of filter input channels
            
            # Check if this is a 2-filter (rotation-invariant) or 4-filter model
            if is_rotation_invariant:
                # 2-filter rotation-invariant model: pad to 4 filters for JS compatibility
                chn = filter_channels // 2
                s = layer[:, :-c].shape  # [n, 2 * chn, fc_dim]
                filter_data = layer[:, :-c].reshape(s[0], chn, 2, s[2])  # [n, chn, 2, fc_dim]
                # Extract identity (index 0) and laplacian (index 1)
                ident = filter_data[:, :, 0:1, :]  # [n, chn, 1, fc_dim]
                lap = filter_data[:, :, 1:2, :]    # [n, chn, 1, fc_dim]
                # Pad to 4 filters: [ident, zero, zero, lap]
                # Zero weights for sobel_x and sobel_y (filter bands 1 and 2)
                zero = np.zeros_like(ident)  # [n, chn, 1, fc_dim]
                filter_data_padded = np.concatenate([
                    ident, zero, zero, lap
                ], axis=2)  # [n, chn, 4, fc_dim]
                # Create new layer array with padded filters
                padded_filters = filter_data_padded.reshape(s[0], chn * 4, s[2])  # [n, 4 * chn, fc_dim]
                # Reconstruct layer with padded filters and bias/pos_emb
                layer = np.concatenate([padded_filters, layer[:, -c:]], axis=1)  # [n, 4 * chn + c, fc_dim]
            else:
                # Standard 4-filter model
                chn = filter_channels // 4
                if filter_channels % 4 != 0:
                    raise ValueError(f"Expected 4-filter model but filter_channels ({filter_channels}) is not divisible by 4")
            
            # Rearrange filter channels (4 filters: id, sobelx, sobely, lap)
            s = layer[:, :-c].shape  # [n, 4 * chn, fc_dim] (after padding if needed)
            # Reconstruct the layer to avoid in-place assignment issues
            rearranged_filters = (layer[:, :-c]
                            .reshape(s[0], chn, 4, s[2])  # [n, chn, 4, fc_dim]
                            .transpose(0, 2, 1, 3)  # [n, 4, chn, fc_dim]
                            .reshape(s))  # [n, 4 * chn, fc_dim]
            layer = np.concatenate([rearranged_filters, layer[:, -c:]], axis=1)  # [n, 4 * chn + c, fc_dim]
            # Update shape to reflect padded dimensions
            shape = [layer.shape[1], layer.shape[2]]  # [c_in, fc_dim] after padding
        
        # Pad channels to be multiple of 4 (WebGL requirement)
        s = layer.shape  # [n, c_in, fc_dim]
        layer = np.pad(layer, ((0, 0), (0, 0), (0, (4 - s[2]) % 4)), mode='constant')
        # After padding: [n, c_in, padded_fc_dim] where padded_fc_dim is multiple of 4
        # Update shape to reflect fc_dim padding
        shape = [shape[0], layer.shape[2]]  # [c_in, padded_fc_dim]
        
        # Reshape for RGBA packing: [n, c_in, padded_fc_dim] -> [n, c_in, padded_fc_dim//4, 4]
        layer = layer.reshape(s[0], s[1], -1, 4)  # [n, c_in, fc_dim//4, 4]
        
        # Tile into 2D grid
        n, ht, wt = layer.shape[:3]  # n=1, ht=c_in, wt=fc_dim//4
        w = 1
        while w < n and w * wt < (n + w - 1) // w * ht:
            w += 1
        layer = tile2d(layer, w)
        layout = (w, (n + w - 1) // w)
        
        # Compute scale and center for quantization
        scale = float(layer.max() - layer.min())
        center = float(-layer.min() / scale) if scale > 0 else 0.0
        
        # Normalize to [0, 1]
        layer_normalized = layer - layer.min()
        if scale > 0:
            layer_normalized = layer_normalized / scale
        else:
            layer_normalized = np.zeros_like(layer_normalized)
        
        # Store normalized float values (not quantized to uint8)
        layer_flatten = layer_normalized.flatten()
        
        # Create layer dictionary
        layer_js = {
            'scale': scale,
            'center': center,
            'data_flatten': list(map(float, layer_flatten)),
            'data_shape': list(layer.shape),  # Final tiled shape
            'shape': list(shape),  # Original shape [c_in, fc_dim]
            'layout': list(layout),
            'pos_emb': (i == 0) and metadata['pos_emb'],
            'bias': (i == 0),  # Only first layer has bias
        }
        models_js['layers'].append(layer_js)
    
    return models_js


def convert_pt_to_json(pt_path, output_path=None, noise_level=None, pos_emb=False):
    """
    Convert a PyTorch .pt file to JSON format for NoiseNCA.
    
    Args:
        pt_path: Path to input .pt file
        output_path: Path to output .json file (default: same name as input)
        noise_level: Noise level value (if not in state_dict)
        pos_emb: Whether the model uses positional embedding
        
    Returns:
        Path to the created JSON file
    """
    print(f"Loading PyTorch model from: {pt_path}")
    state_dict = torch.load(pt_path, map_location='cpu')
    
    # Convert to numpy format and detect rotation-invariant
    np_params, is_rotation_invariant = torch_model_to_np(state_dict)
    
    # Override noise_level if provided
    if noise_level is not None:
        np_params[-1] = noise_level
    
    # Extract model name from filename
    model_name = os.path.splitext(os.path.basename(pt_path))[0]
    
    # Export to JSON format
    metadata = {
        'model_names': [model_name],
        'pos_emb': pos_emb,
        'rotation_invariant': is_rotation_invariant
    }
    
    js_models = export_np_models_to_json(np_params, metadata)
    
    # Determine output path
    if output_path is None:
        output_path = os.path.splitext(pt_path)[0] + '.json'
    
    # Save JSON
    print(f"Saving JSON to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(js_models, f, indent=2)
    
    print("Conversion complete!")
    print(f"  - Model name: {model_name}")
    print(f"  - Layers: {len(js_models['layers'])}")
    print(f"  - Noise level: {js_models['noise_level']}")
    print(f"  - Positional embedding: {pos_emb}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Convert PyTorch .pt model files to JSON format for NoiseNCA demo'
    )
    parser.add_argument('input', help='Input .pt file path')
    parser.add_argument('output', nargs='?', help='Output .json file path (optional)')
    parser.add_argument('--noise-level', type=float, help='Noise level value (if not in model)')
    parser.add_argument('--pos-emb', action='store_true', 
                       help='Model uses positional embedding (default: False)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    try:
        convert_pt_to_json(args.input, args.output, args.noise_level, args.pos_emb)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
