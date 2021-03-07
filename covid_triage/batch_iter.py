import numpy as np
from dpipe.checks import check_shapes


def get_sampling_weights(ids, id_to_label, label_to_weight):
    labels = [id_to_label[i] for i in ids]
    label_to_count = dict(zip(*np.unique(labels, return_counts=True)))
    return [label_to_weight[l] / label_to_count[l] for l in labels]


def sample_mask_from_dict(d: dict, shape=None, keys=None):
    keys = list(set(d.keys()) & set(keys)) if keys is not None else list(d.keys())

    if not keys:
        assert shape is not None
        return np.full(shape, np.nan)

    return d[keys[np.random.randint(len(keys))]]


def drop_slices(*arrays, n_slices_left, n_scalars=0):
    if n_scalars:
        arrays, scalars = arrays[:-n_scalars], arrays[-n_scalars:]
    else:
        scalars = []

    check_shapes(*arrays)
    n_slices = arrays[0].shape[-1]

    if n_slices > n_slices_left:
        stride, remainder = divmod(n_slices, n_slices_left)
        shift = np.random.randint(stride + remainder)
        arrays = [a[..., shift::stride] for a in arrays]

    return (*arrays, *scalars)
