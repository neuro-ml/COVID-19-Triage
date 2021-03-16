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


def save_image(root, id_, image, spacing):
    root = Path(root)
    path = root / 'images' / id_
    path.mkdir(parents=True, exist_ok=True)
    save(image, path / 'image.npy')
    save(spacing, path / 'spacing.json')


def save_lungs_mask(root, id_, lungs_mask):
    root = Path(root)
    path = root / 'lungs_masks' / id_
    path.mkdir(parents=True, exist_ok=True)
    save(lungs_mask, path / f'lungs_mask.npy')


def save_covid_mask(root, id_, rater, covid_mask):
    root = Path(root)
    path = root / 'covid_masks' / id_ / rater
    path.mkdir(parents=True, exist_ok=True)
    save(covid_mask, path / f'covid_mask.npy')


def save_covid_label(root, id_, covid_label):
    root = Path(root)
    path = root / 'covid_labels' / id_
    path.mkdir(parents=True, exist_ok=True)
    save(covid_label, path / f'covid_label.json')


def save_subset(root, name, ids):
    root = Path(root)
    path = root / 'subsets' / name
    path.mkdir(parents=True, exist_ok=True)
    save(ids, path / 'ids.json')
