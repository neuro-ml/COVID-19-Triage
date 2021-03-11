from pathlib import Path
import numpy as np
import pydicom
import nibabel
from dpipe.io import load, save


def load_dicoms(root):
    from pydicom.errors import InvalidDicomError
    
    dicoms = []
    for file in Path(root).glob('**/*'):
        if not file.is_file():
            continue
            
        try:
            dicoms.append(pydicom.dcmread(file))
        except InvalidDicomError:
            continue
            
    return dicoms


def load_nii_arr(file):
    arr = nibabel.as_closest_canonical(nibabel.load(file)).get_fdata()
    arr = np.swapaxes(arr[::-1, ::-1, ::-1], 0, 1)
    return arr


def load_nii_spacing(file):
    zooms = nibabel.as_closest_canonical(nibabel.load(file)).header.get_zooms()
    return zooms[1], zooms[0], zooms[2]


def save_image(root, id_, image, spacing):
    path = root / 'images' / id_
    path.mkdir(parents=True, exist_ok=True)
    save(image, path / 'image.npy')
    save(spacing, path / 'spacing.json')


def save_lungs_mask(root, id_, lungs_mask):
    path = root / 'lungs_masks' / id_
    path.mkdir(parents=True, exist_ok=True)
    save(lungs_mask, path / f'lungs_mask.npy')


def save_covid_mask(root, id_, rater, covid_mask):
    path = root / 'covid_masks' / id_ / rater
    path.mkdir(parents=True, exist_ok=True)
    save(covid_mask, path / f'covid_mask.npy')


def save_covid_label(root, id_, covid_label):
    path = root / 'covid_labels' / id_
    path.mkdir(parents=True, exist_ok=True)
    save(covid_label, path / f'covid_label.json')
