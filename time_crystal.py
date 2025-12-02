#!/usr/bin/env python3
"""
Time crystal analysis for NCA models.
Analyzes long-time dynamics of noisy CA systems.
"""

import argparse
import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Fix for matplotlib compatibility issue
try:
    import matplotlib.cbook
    if not hasattr(matplotlib.cbook, "_Stack"):
        class _Stack(list):
            def push(self, item):
                self.append(item)
                return item
            def pop(self):
                return super().pop() if self else None
            def current(self):
                return self[-1] if self else None
        matplotlib.cbook._Stack = _Stack
except:
    pass

from models import NCA, NoiseNCA, PENCA


def get_nca_model(config, texture_name, device):
    """Create an NCA model instance based on config and texture name."""
    model_type = config['model']['type']
    attr = config['model']['attr'].copy()
    attr['device'] = device
    
    if model_type == 'NCA':
        return NCA(**attr)
    elif model_type == 'NoiseNCA':
        noise_levels = config['model']['noise_levels']
        if texture_name in noise_levels:
            noise_level = noise_levels[texture_name]
        else:
            noise_level = noise_levels['default']
        return NoiseNCA(noise_level=noise_level, **attr)
    elif model_type == 'PENCA':
        return PENCA(**attr)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def compute_pixel_distance(s1, s2):
    """
    Compute pixel-wise distance between two states using only RGB channels.
    Same metric as used in lyapunov.py.
    
    Args:
        s1, s2: State tensors of shape [b, chn, h, w]
    
    Returns:
        Average pixel-wise distance (scalar) computed using only the first 3 channels (RGB)
    """
    # Compute L2 distance per pixel using only RGB channels (first 3 channels)
    diff = s1[:, :3, :, :] - s2[:, :3, :, :]  # Only use RGB channels
    pixel_distances = torch.norm(diff, dim=1)  # [b, h, w] - L2 norm over RGB channels
    return pixel_distances.mean().item()


def run_distance_mode(model, device, height, width, dx, dy, dt, epsilon,
                      tquench, tevolve, n_runs):
    """
    Run distance mode: compute pixel-wise L2 distance from reference state s0.
    
    Args:
        model: NCA model
        device: PyTorch device
        height, width: Grid dimensions
        dx, dy: Spatial scaling parameters
        dt: Time step
        epsilon: Per-step noise strength
        tquench: Time to evolve before starting measurement
        tevolve: Time to evolve while measuring
        n_runs: Number of independent runs
    
    Returns:
        times: Array of time steps after tquench
        distances: Array of averaged pixel-wise distances (averaged over n_runs)
        s0_rgb: RGB image at time tquench
    """
    model.eval()
    
    # Calculate steps
    steps_quench = int(tquench / dt)
    steps_evolve = int(tevolve / dt)
    
    print(f"Quench: {steps_quench} steps (t={tquench})")
    print(f"Evolve: {steps_evolve} steps (t={tevolve})")
    
    # Step 1: Quench phase - evolve to reference state s0
    print("\nQuenching to reference state...")
    with torch.no_grad():
        s = model.seed(1, height, width).to(device)
        for step in tqdm(range(steps_quench), desc="Quenching"):
            s = model(s, dx=dx, dy=dy, dt=dt)
            # Apply per-step noise if epsilon > 0
            if epsilon > 0:
                noise = (torch.rand_like(s) - 0.5) * epsilon
                s = s + noise
        
        # Save reference state s0 (full state for evolution, RGB for display)
        s0 = s.clone()  # [1, chn, h, w] - full state
        
        # Convert to numpy for display (RGB only)
        s0_rgb = model.to_rgb(s0).permute(0, 2, 3, 1).cpu().numpy()[0]
        s0_rgb = np.clip(s0_rgb, 0, 1)
    
    # Display reference image
    plt.figure(figsize=(8, 8))
    plt.imshow(s0_rgb)
    plt.title(f'Reference State at t={tquench}')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    # Step 2: Evolution phase - compute distances for n_runs
    print(f"\nRunning {n_runs} independent trials...")
    all_distances = []
    
    for run in range(n_runs):
        print(f"\nRun {run + 1}/{n_runs}")
        distances_run = []
        
        with torch.no_grad():
            # Start from same reference state s0 (but will diverge due to noise)
            s = s0.clone()
            
            for step in tqdm(range(steps_evolve), desc=f"Run {run + 1}"):
                # Evolve state
                s = model(s, dx=dx, dy=dy, dt=dt)
                
                # Apply per-step noise if epsilon > 0
                if epsilon > 0:
                    noise = (torch.rand_like(s) - 0.5) * epsilon
                    s = s + noise
                
                # Compute distance from reference s0
                # s is [1, 3, h, w] (RGB only from s0)
                # s0 is [1, 3, h, w]
                dist = compute_pixel_distance(s, s0)
                distances_run.append(dist)
        
        all_distances.append(distances_run)
    
    # Average over runs
    all_distances = np.array(all_distances)  # [n_runs, steps_evolve]
    distances = np.mean(all_distances, axis=0)  # [steps_evolve]
    
    # Create time array
    times = np.arange(steps_evolve) * dt
    
    return times, distances, s0_rgb


def main():
    parser = argparse.ArgumentParser(description='Time crystal analysis for NCA models')
    parser.add_argument('--mode', type=str, default='distance',
                        choices=['distance'],
                        help='Analysis mode')
    parser.add_argument('--model_type', type=str, default='Noise-NCA',
                        choices=['Noise-NCA', 'PE-NCA', 'Vanilla-NCA'],
                        help='Type of model')
    parser.add_argument('--texture', type=str, default='spiralled_0124',
                        help='Texture name')
    parser.add_argument('--dx', type=float, default=1.0,
                        help='Spatial scaling in x direction (default: 1.0)')
    parser.add_argument('--dt', type=float, default=0.1,
                        help='Time step (default: 0.1)')
    parser.add_argument('--epsilon', type=float, default=0.0,
                        help='Per-step noise strength (default: 0.0)')
    parser.add_argument('--height', type=int, default=96,
                        help='Grid height (default: 128)')
    parser.add_argument('--width', type=int, default=96,
                        help='Grid width (default: 128)')
    parser.add_argument('--n_runs', type=int, default=5,
                        help='Number of independent runs (default: 5)')
    parser.add_argument('--tquench', type=float, default=100.0,
                        help='Burn-in time before measurement (default: 100.0)')
    parser.add_argument('--tevolve', type=float, default=500.0,
                        help='Time to evolve while measuring (default: 500.0)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output plot filename (default: auto-generated)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (default: cuda if available, else cpu)')
    
    args = parser.parse_args()
    
    # Load config
    config_map = {
        'Noise-NCA': 'configs/Noise-NCA.yml',
        'PE-NCA': 'configs/PE-NCA.yml',
        'Vanilla-NCA': 'configs/Vanilla-NCA.yml'
    }
    
    config_path = config_map[args.model_type]
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load model
    model_path = os.path.join('trained_models', args.model_type, args.texture, 'weights.pt')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found: {model_path}")
    
    device = torch.device(args.device)
    model = get_nca_model(config, args.texture, device)
    
    print(f"Loading model from: {model_path}")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    
    print(f"\nTime Crystal Analysis - Mode: {args.mode}")
    print(f"Parameters:")
    print(f"  Texture: {args.texture}")
    print(f"  Grid size: {args.height}x{args.width}")
    print(f"  dx={args.dx}, dt={args.dt}, epsilon={args.epsilon}")
    print(f"  tquench={args.tquench}, tevolve={args.tevolve}")
    print(f"  n_runs={args.n_runs}\n")
    
    # Run analysis based on mode
    if args.mode == 'distance':
        times, distances, s0_rgb = run_distance_mode(
            model, device, args.height, args.width,
            args.dx, args.dx, args.dt, args.epsilon,
            args.tquench, args.tevolve, args.n_runs
        )
        
        # Plot results
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Left: Reference image
        axes[0].imshow(s0_rgb)
        axes[0].set_title(f'Reference State at t={args.tquench}')
        axes[0].axis('off')
        
        # Right: Distance vs time
        axes[1].plot(times, distances, 'b-', linewidth=2)
        axes[1].set_xlabel('Time after tquench')
        axes[1].set_ylabel('Pixel-wise L2 Distance from s0')
        axes[1].set_title('Distance from Reference State')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        if args.output is None:
            output_path = f"time_crystal_{args.mode}_{args.texture}_eps{args.epsilon:.3f}.png"
        else:
            output_path = args.output
        plt.savefig(output_path, dpi=150)
        print(f"\nSaved plot to {output_path}")
        
        plt.show()
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == '__main__':
    main()

