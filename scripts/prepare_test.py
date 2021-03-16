import os
import argparse
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import numpy as np
import nibabel

from covid_triage.io import save_image, save_covid_label, save_covid_mask, save_subset


def load_nii_arr(file):
    arr = nibabel.load(file).get_fdata()
    return arr


def load_nii_spacing(file):
    return nibabel.load(file).header.get_zooms()


def main(src, dst):
    src = Path(src)
    dst = Path(dst)

    for file in tqdm(list(src.glob('*/images/*.nii.gz'))):
        id_ = file.name[:-len('.nii.gz')]
        image = load_nii_arr(file).astype(np.int16)
        spacing = load_nii_spacing(file)
        covid_label = file.parents[1].name == 'covid'
        save_image(dst, id_, image, spacing)
        save_covid_label(dst, id_, covid_label)

    for file in src.glob('covid/masks/*/*.nii.gz'):
        id_ = file.parent.name
        rater = file.name[:-len('.nii.gz')]
        covid_mask = load_nii_arr(file).astype(bool)
        save_covid_mask(dst, id_, rater, covid_mask)

    for name in ['covid', 'bacterial_pneumonia', 'nodules', 'normal']:
        ids = [file.name[:-len('.nii.gz')] for file in src.glob(f'{name}/images/*.nii.gz')]
        save_subset(dst, name, ids)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input')
    parser.add_argument('-o', '--output')
    args = parser.parse_args()

    main(args.input, args.output)
