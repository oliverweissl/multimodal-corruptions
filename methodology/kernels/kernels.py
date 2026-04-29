import numpy as np

# IMPORTANT: this implementation is based on PerturbationDrive https://github.com/ast-fortiss-tum/perturbation-drive

def create_disk_kernel(radius):
    """Create a disk-shaped kernel with the given radius."""
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    mask = x**2 + y**2 <= radius**2
    kernel = np.zeros((2 * radius + 1, 2 * radius + 1), dtype=np.float32)
    kernel[mask] = 1
    # Normalize the kernel so that the sum of its elements is 1.
    kernel /= kernel.sum()
    return kernel

def create_motion_blur_kernel(size, angle):
    """
    Create a motion blur kernel of the given size and angle.
    """
    # Create an empty kernel
    kernel = np.zeros((size, size))
    # Convert angle to radian
    angle = np.deg2rad(angle)
    # Calculate the center of the kernel
    center = size // 2
    # Calculate the slope of the line
    slope = np.tan(angle)
    # Fill in the kernel
    for y in range(size):
        x = int(slope * (y - center) + center)
        if 0 <= x < size:
            kernel[y, x] = 1
    # Normalize the kernel
    kernel /= kernel.sum()
    return kernel

