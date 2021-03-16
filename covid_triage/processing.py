import numpy as np
import warnings
from dpipe.im import describe_connected_components, crop_to_shape, crop_to_box, \
    zoom_to_shape, restore_crop
from dpipe.im.box import get_volume as get_box_volume, Box
from dpipe.itertools import extract

from sklearn.cluster import KMeans


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


def crop_to_max_shape_xy(x, max_shape_xy, axes_xy=(0, 1)):
    shape = np.minimum(extract(x.shape, axes_xy), max_shape_xy)
    return crop_to_shape(x, shape, axes_xy)


def zoom_box(box: Box, scale_factor):
    start, stop = box * scale_factor
    return np.array([np.ceil(start), np.floor(stop)]).astype(int)


def zoom_to_shape_within_box(x, shape, box: Box, padding_values=0):
    assert x.ndim == box.shape[-1] == len(shape)
    assert np.all(0 <= box[0]) and np.all(box[0] <= box[1]) and np.all(box[1] <= x.shape), \
        '``box`` is invalid for ``x``.'

    zoomed_box = zoom_box(box, np.asarray(shape) / x.shape)
    if get_box_volume(zoomed_box) == 0:
        warnings.warn('Zoomed box has zero size.')
        return np.zeros(shape, dtype=x.dtype)

    zoomed_x_within_box = zoom_to_shape(crop_to_box(x, box), zoomed_box[1] - zoomed_box[0])
    return restore_crop(zoomed_x_within_box, zoomed_box, shape, padding_values=padding_values)


def separate_lungs(lung_mask: np.ndarray, spacing: [float, tuple] = 1., step: int = 30, seed=42) -> np.ndarray:
    """
    Assigns labels 1, 2 to the lung lobes.
    Label 1 is assigned to the lobe with a smaller coordinate along frontal axis.
    """
    spacing = np.broadcast_to(spacing, lung_mask.ndim)

    labeled_mask = np.zeros(lung_mask.shape, dtype=int)

    if not np.any(lung_mask):
        return labeled_mask

    indices = np.stack(np.nonzero(lung_mask), axis=-1)
    point_cloud = indices * spacing

    if len(point_cloud) <= 2 * step:
        step = 1

    kmeans = KMeans(n_clusters=2, random_state=seed)
    kmeans.fit(point_cloud[::step])
    predicted_labels = kmeans.predict(point_cloud)

    frontal_axis = kmeans.cluster_centers_.std(axis=0).argmax()

    for lung_label, kmeans_label in zip((1, 2), kmeans.cluster_centers_[:, frontal_axis].argsort()):
        labeled_mask[tuple(indices[predicted_labels == kmeans_label].T)] = lung_label

    return labeled_mask
