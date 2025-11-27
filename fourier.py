#!/usr/bin/env python3
"""
Fourier analysis script for NCA models.
Computes the 2-point correlation function (averaged over pixels) as a function of time.

Usage:
    python fourier.py --model_type Noise-NCA --texture bubbly_0101 --tquench 100 --tevolve 500
"""

import argparse
import os
import sys
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt

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
        return NoiseNCA(noise_level=1.0, **attr)
    elif model_type == 'PENCA':
        return PENCA(**attr)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def rgb_to_grayscale(img):
    """Convert RGB image to grayscale."""
    if len(img.shape) == 3:
        return np.mean(img, axis=2)
    return img


def compute_two_point_correlation(img):
    """
    Compute the spatially-averaged 2-point correlation function C(r).
    
    C(r) = <phi(x) * phi(x+r)>_x - <phi>^2
    
    This is computed efficiently via FFT:
    C(r) = IFFT(|FFT(phi)|^2) / N - mean^2
    
    Returns the angular average as a function of radial distance r.
    
    Args:
        img: Image array of shape (H, W, 3) or (H, W) with values in [0, 1]
    
    Returns:
        r: Array of radial distances
        C_r: Angular averaged correlation function
    """
    gray = rgb_to_grayscale(img)
    
    # Subtract mean to get fluctuations
    mean_val = np.mean(gray)
    phi = gray - mean_val
    
    # Compute autocorrelation via FFT
    fft = np.fft.fft2(phi)
    power = np.abs(fft) ** 2
    autocorr = np.fft.ifft2(power)
    autocorr = np.real(autocorr) / phi.size  # Normalize by number of pixels
    
    # Shift zero lag to center
    autocorr_shifted = np.fft.fftshift(autocorr)
    
    # Compute angular average
    r, C_r = compute_angular_average_2d(autocorr_shifted)
    
    return r, C_r


def compute_angular_average_2d(data):
    """Compute angular average of a 2D array (radial average)."""
    h, w = data.shape
    center_y, center_x = h // 2, w // 2
    
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    r = r.astype(int)
    
    r_flat = r.flatten()
    data_flat = data.flatten()
    
    r_max = int(np.max(r_flat))
    
    values = []
    radii = []
    for r_val in range(r_max + 1):
        mask = (r_flat == r_val)
        if np.any(mask):
            avg_value = np.mean(data_flat[mask])
            values.append(avg_value)
            radii.append(r_val)
    
    return np.array(radii), np.array(values)


def main():
    parser = argparse.ArgumentParser(description='Compute 2-point correlation function over time')
    parser.add_argument('--model_type', type=str, required=True,
                        choices=['Noise-NCA', 'PE-NCA', 'Vanilla-NCA'],
                        help='Type of model')
    parser.add_argument('--texture', type=str, required=True,
                        help='Texture name')
    parser.add_argument('--tquench', type=float, default=100,
                        help='Time to evolve before starting measurement (default: 100)')
    parser.add_argument('--tevolve', type=float, default=500,
                        help='Time to evolve while measuring (default: 500)')
    parser.add_argument('--dt', type=float, default=0.1,
                        help='Time step (default: 0.1)')
    parser.add_argument('--height', type=int, default=128,
                        help='Height of simulation (default: 128)')
    parser.add_argument('--width', type=int, default=128,
                        help='Width of simulation (default: 128)')
    parser.add_argument('--sample_interval', type=float, default=10.0,
                        help='Time interval between correlation measurements (default: 10.0)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    
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
    model.eval()
    
    # Calculate steps
    quench_steps = int(args.tquench / args.dt)
    evolve_steps = int(args.tevolve / args.dt)
    sample_steps = int(args.sample_interval / args.dt)
    
    print(f"Quench steps: {quench_steps}, Evolve steps: {evolve_steps}, Sample every {sample_steps} steps")
    
    # Initialize state
    with torch.no_grad():
        s = model.seed(1, args.height, args.width).to(device)
        
        # Quench phase - let system equilibrate
        print(f"Quenching for t={args.tquench}...")
        for step in range(quench_steps):
            s[:] = model(s, dt=args.dt)
            if (step + 1) % (quench_steps // 10) == 0:
                print(f"  Quench progress: {100 * (step + 1) / quench_steps:.0f}%")
        
        # Measurement phase
        print(f"Measuring correlation for t={args.tevolve}...")
        times = []
        correlations = []  # List of (r, C_r) tuples
        
        for step in range(evolve_steps):
            s[:] = model(s, dt=args.dt)
            
            # Sample correlation function
            if step % sample_steps == 0:
                img = model.to_rgb(s[0]).permute(1, 2, 0).cpu().numpy()
                img = np.clip(img, 0, 1)
                
                r, C_r = compute_two_point_correlation(img)
                times.append(step * args.dt)
                correlations.append((r, C_r))
                
                if len(times) % 10 == 0:
                    print(f"  Collected {len(times)} samples, t={times[-1]:.1f}")
    
    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left plot: C(r) at different times (subset)
    ax1 = axes[0]
    num_curves = min(10, len(correlations))
    indices = np.linspace(0, len(correlations) - 1, num_curves, dtype=int)
    cmap = plt.cm.viridis
    
    for i, idx in enumerate(indices):
        r, C_r = correlations[idx]
        t = times[idx]
        # Limit to half the image width
        max_r = args.width // 2
        mask = r <= max_r
        color = cmap(i / (num_curves - 1)) if num_curves > 1 else cmap(0.5)
        ax1.plot(r[mask], C_r[mask], color=color, label=f't={t:.0f}')
    
    ax1.set_xlabel('r (pixels)')
    ax1.set_ylabel('C(r)')
    ax1.set_title('2-Point Correlation Function')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Right plot: C(r=0) and C(r=r_char) vs time
    ax2 = axes[1]
    
    # Extract C(0) over time (variance of the field)
    C_0 = [C_r[0] for r, C_r in correlations]
    
    # Extract C at a characteristic length scale (e.g., r=10)
    r_char = 10
    C_r_char = []
    for r, C_r in correlations:
        if r_char < len(C_r):
            C_r_char.append(C_r[r_char])
        else:
            C_r_char.append(np.nan)
    
    ax2.plot(times, C_0, 'b-', linewidth=2, label='C(r=0)')
    ax2.plot(times, C_r_char, 'r-', linewidth=2, label=f'C(r={r_char})')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Correlation')
    ax2.set_title('Correlation vs Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_path = f"correlation_{args.texture}.png"
    plt.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")
    
    plt.show()


if __name__ == '__main__':
    main()

