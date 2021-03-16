import numpy as np
from dpipe.im.utils import get_mask_volume
from .processing import separate_lungs


def invert_stratification_mapping(label_to_ids):
    assert sum(map(len, label_to_ids.values())) == len(set().union(*map(set, label_to_ids.values()))), \
        'Some ids have more than one stratification label.'

    id_to_label = dict()
    for label, ids in label_to_ids.items():
        id_to_label.update({id_: label for id_ in ids})

    return id_to_label


def get_lungs_and_lesions_volumes(lungs_mask: np.ndarray, lesions_mask: np.ndarray,
                                  spacing) -> tuple:
    """ Volumes order: bigger lung, smaller lung. """
    assert lungs_mask.dtype == bool

    lungs_mask = separate_lungs(lungs_mask)

    assert np.issubdtype(lungs_mask.dtype, np.int64), lungs_mask.dtype
    assert len(np.unique(lungs_mask)) == 3, np.unique(lungs_mask)
    assert lesions_mask.dtype == bool, lesions_mask.dtype

    lungs_mask = np.array([lungs_mask == 1, lungs_mask == 2])
    lungs_volumes = np.array([get_mask_volume(m, *spacing) for m in lungs_mask])

    lesions_mask = lesions_mask[None] & lungs_mask
    lesions_volumes = np.array([get_mask_volume(m, *spacing) for m in lesions_mask])

    if lungs_volumes[1] > lungs_volumes[0]:
        lungs_volumes = lungs_volumes[::-1]
        lesions_volumes = lesions_volumes[::-1]

    return lungs_volumes, lesions_volumes
