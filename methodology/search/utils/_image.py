import numpy as np
from PIL import Image


def ensure_rgb(img: np.ndarray | Image.Image) -> np.ndarray:
    """Convert any image-like input to a 3-channel uint8 numpy array.

    :param img: PIL image or numpy array (grayscale, single-channel, or RGB).
    :returns: HxWx3 numpy array; passthrough if already 3-channel.
    """
    if isinstance(img, Image.Image):
        img = np.array(img)
    if img.ndim == 3 and img.shape[2] == 3:
        return img
    if img.ndim == 2:
        return np.stack([img] * 3, axis=-1)
    if img.ndim == 3 and img.shape[2] == 1:
        return np.repeat(img, 3, axis=2)
    return img


def resize_image_smart(img: Image.Image, max_side: int = 1080) -> Image.Image:
    """Proportionally downscale ``img`` so its longest side is at most ``max_side``.

    :param img: Input PIL image.
    :param max_side: Maximum pixel length for the longest side.
    :returns: Downscaled PIL image, or the original if already within bounds.
    """
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    scale = max_side / max(w, h)
    return img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
