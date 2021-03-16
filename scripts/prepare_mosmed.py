import os
import argparse
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import numpy as np
import nibabel

from covid_triage.io import save_image, save_covid_label, save_covid_mask


def load_nii_arr(file):
    arr = nibabel.as_closest_canonical(nibabel.load(file)).get_fdata()
    arr = np.swapaxes(arr[::-1, ::-1, ::-1], 0, 1)
    return arr


def load_nii_spacing(file):
    zooms = nibabel.as_closest_canonical(nibabel.load(file)).header.get_zooms()
    return zooms[1], zooms[0], zooms[2]


def main(src, dst):
    src = Path(src)
    dst = Path(dst)
    for file in tqdm(list(src.glob('ct*/*.nii.gz'))):
        id_ = file.name[:-len('.nii.gz')]
        image = load_nii_arr(file).astype(np.int16)
        spacing = load_nii_spacing(file)

        save_image(dst, id_, image, spacing)

        covid_mask_file = src / 'masks' / f'{id_}_mask.nii.gz'
        if covid_mask_file.exists():
            covid_mask = load_nii_arr(covid_mask_file).astype(bool)
            assert covid_mask.shape == image.shape
            save_covid_mask(dst, id_, 'mosmed', covid_mask)

        covid_label = int(file.parent.name[-1]) > 0
        save_covid_label(dst, id_, covid_label)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input')
    parser.add_argument('-o', '--output')
    args = parser.parse_args()

    main(args.input, args.output)