import os
import argparse
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import numpy as np
import nibabel

from covid_triage.io import save_image, save_covid_label, \
    save_covid_mask, save_lungs_mask


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

    rp_ids = list(map(str, range(1, 10))) 
    coronacases_ids = [f'coronacases_{i:03d}' for i in range(1, 11)]
    radiopaedia_ids = [
        'radiopaedia_10_85902_1',
        'radiopaedia_10_85902_3',
        'radiopaedia_14_85914_0',
        'radiopaedia_27_86410_0',
        'radiopaedia_29_86490_1',
        'radiopaedia_29_86491_1',
        'radiopaedia_36_86526_0',
        'radiopaedia_40_86625_0',
        'radiopaedia_4_85506_1',
        'radiopaedia_7_85703_0'
    ]
    ids = rp_ids + coronacases_ids + radiopaedia_ids
    for id_ in tqdm(ids):
        if id_ in rp_ids:
            image_file = src / 'rp_im' / f'{id_}.nii.gz'
            covid_mask_file = src / 'rp_msk' / f'{id_}.nii.gz'
            lungs_mask_file = src / 'rp_lung_msk' / f'{id_}.nii.gz'
        else:
            image_file = src / 'COVID-19-CT-Seg_20cases' / f'{id_}.nii.gz'
            covid_mask_file = src / 'Infection_Mask' / f'{id_}.nii.gz'
            lungs_mask_file = src / 'Lung_Mask' / f'{id_}.nii.gz'

        image = load_nii_arr(image_file).astype(np.int16)
        spacing = load_nii_spacing(image_file)
        covid_mask = load_nii_arr(covid_mask_file) > 0
        lungs_mask = load_nii_arr(lungs_mask_file) > 0
        
        if id_.startswith('radiopaedia'):
            image = image[::-1]
            covid_mask = covid_mask[::-1]
            lungs_mask = lungs_mask[::-1]

        if id_ == 'radiopaedia_7_85703_0':
            image = image[..., ::-1]
            covid_mask = covid_mask[..., ::-1]
            lungs_mask = lungs_mask[..., ::-1]

        assert covid_mask.shape == image.shape
        assert lungs_mask.shape == image.shape

        save_image(dst, id_, image, spacing)
        save_covid_mask(dst, id_, 'medseg', covid_mask)
        save_lungs_mask(dst, id_, lungs_mask)
        save_covid_label(dst, id_, covid_label=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input')
    parser.add_argument('-o', '--output')
    args = parser.parse_args()

    main(args.input, args.output)
