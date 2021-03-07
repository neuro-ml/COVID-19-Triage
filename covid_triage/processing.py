import numpy as np


def is_uint8(image):
    return np.allclose(image, image.astype(np.uint8).astype(image.dtype))


def normalize_intensities(image, low: float, high: float):
    assert low < high, f'``low`` {low} must be less than ``high`` {high}.'

    if is_uint8(image):
        image = image.astype(np.float32) / 255
    else:
        image = image.astype(np.float32)
        image = np.clip(image, low, high)
        image = (image - low) / (high - low)
        
    return image
