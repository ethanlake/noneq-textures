"""
Generate synthetic texture images for NCA training.
"""

import numpy as np
from PIL import Image
import argparse
import os


def generate_vertical_stripes(size=256, num_stripes=10, smooth=False):
    """Generate an image with uniform vertical black and white stripes.
    
    num_stripes should be even to have equal black and white stripes.
    """
    img = np.zeros((size, size, 3), dtype=np.uint8)
    
    for x in range(size):
        if smooth:
            # Sinusoidal variation: 0.5 + 0.5*cos gives [0, 1]
            val = 0.5 + 0.5 * np.cos(2 * np.pi * x * num_stripes / size)
            img[:, x, :] = int(val * 255)
        else:
            stripe_idx = int(x * num_stripes / size)
            if stripe_idx % 2 == 0:
                img[:, x, :] = 255
    
    return img


def generate_horizontal_stripes(size=256, num_stripes=10, smooth=False):
    """Generate an image with uniform horizontal black and white stripes."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    
    for y in range(size):
        if smooth:
            val = 0.5 + 0.5 * np.cos(2 * np.pi * y * num_stripes / size)
            img[y, :, :] = int(val * 255)
        else:
            stripe_idx = int(y * num_stripes / size)
            if stripe_idx % 2 == 0:
                img[y, :, :] = 255
    
    return img


def generate_diagonal_stripes(size=256, num_stripes=10, smooth=False):
    """Generate an image with uniform diagonal (45 degree) black and white stripes."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    
    for y in range(size):
        for x in range(size):
            diag_pos = (x + y) / (2 * size - 2)
            if smooth:
                val = 0.5 + 0.5 * np.cos(2 * np.pi * diag_pos * num_stripes)
                img[y, x, :] = int(val * 255)
            else:
                stripe_idx = int(diag_pos * num_stripes)
                if stripe_idx % 2 == 0:
                    img[y, x, :] = 255
    
    return img


def generate_bullseye(size=256, num_stripes=10, smooth=False):
    """Generate an image with concentric black and white rings (bullseye pattern)."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    
    center = size / 2
    max_radius = size / 2 * np.sqrt(2)
    
    for y in range(size):
        for x in range(size):
            r = np.sqrt((x - center + 0.5)**2 + (y - center + 0.5)**2)
            if smooth:
                val = 0.5 + 0.5 * np.cos(2 * np.pi * r * num_stripes / max_radius)
                img[y, x, :] = int(val * 255)
            else:
                ring_idx = int(r * num_stripes / max_radius)
                if ring_idx % 2 == 0:
                    img[y, x, :] = 255
    
    return img


def generate_burst(size=256, num_rays=10, smooth=False):
    """Generate an image with alternating black and white rays focusing to the center.
    
    num_rays should be even to have equal black and white rays.
    """
    img = np.zeros((size, size, 3), dtype=np.uint8)
    
    center = size / 2
    
    for y in range(size):
        for x in range(size):
            # Calculate angle from center (in radians)
            dx = x - center + 0.5
            dy = y - center + 0.5
            angle = np.arctan2(dy, dx)
            # Normalize angle to [0, 2π]
            if angle < 0:
                angle += 2 * np.pi
            
            if smooth:
                # Sinusoidal variation based on angle
                val = 0.5 + 0.5 * np.cos(angle * num_rays)
                img[y, x, :] = int(val * 255)
            else:
                # Map angle to ray index
                ray_idx = int(angle * num_rays / (2 * np.pi)) % num_rays
                if ray_idx % 2 == 0:
                    img[y, x, :] = 255
    
    return img


def generate_french_flag(size=256, num_stripes=None, smooth=False):
    """Generate a French flag pattern with three vertical stripes: blue, white, red."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    
    # French flag colors (RGB)
    blue = np.array([0, 85, 164], dtype=np.uint8)
    white = np.array([255, 255, 255], dtype=np.uint8)
    red = np.array([239, 65, 53], dtype=np.uint8)
    
    # Divide width into three equal parts
    third = size / 3
    
    for y in range(size):
        for x in range(size):
            if x < third:
                # Left third: blue
                img[y, x, :] = blue
            elif x < 2 * third:
                # Middle third: white
                img[y, x, :] = white
            else:
                # Right third: red
                img[y, x, :] = red
    
    return img


def save_texture(img, filename, quality=95):
    """Save image as JPEG with specified quality."""
    pil_img = Image.fromarray(img)
    pil_img.save(filename, 'JPEG', quality=quality)


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic texture images')
    parser.add_argument('--pattern', type=str, default='vertical_stripes',
                        choices=['vertical_stripes', 'horizontal_stripes', 'diagonal_stripes', 'bullseye', 'burst', 'french_flag'],
                        help='Pattern type to generate')
    parser.add_argument('--output', type=str, default='generated_texture.jpg',
                        help='Output filename')
    parser.add_argument('--size', type=int, default=256,
                        help='Image size (square)')
    parser.add_argument('--num_stripes', type=int, default=10,
                        help='Number of stripes (for stripe patterns)')
    parser.add_argument('--smooth', action='store_true',
                        help='Use sinusoidal variation instead of sharp stripes')
    
    args = parser.parse_args()
    
    if args.pattern == 'vertical_stripes':
        img = generate_vertical_stripes(size=args.size, num_stripes=args.num_stripes, smooth=args.smooth)
    elif args.pattern == 'horizontal_stripes':
        img = generate_horizontal_stripes(size=args.size, num_stripes=args.num_stripes, smooth=args.smooth)
    elif args.pattern == 'diagonal_stripes':
        img = generate_diagonal_stripes(size=args.size, num_stripes=args.num_stripes, smooth=args.smooth)
    elif args.pattern == 'bullseye':
        img = generate_bullseye(size=args.size, num_stripes=args.num_stripes, smooth=args.smooth)
    elif args.pattern == 'burst':
        img = generate_burst(size=args.size, num_rays=args.num_stripes, smooth=args.smooth)
    elif args.pattern == 'french_flag':
        img = generate_french_flag(size=args.size, smooth=args.smooth)
    
    save_texture(img, args.output)
    print(f"Generated {args.pattern} texture: {args.output}")


if __name__ == '__main__':
    main()

