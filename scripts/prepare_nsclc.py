import os
import argparse
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import numpy as np
import nibabel

from dicom_csv import get_common_tag, get_slices_plane, Plane, order_series, stack_images, get_voxel_spacing
from dicom_csv.exceptions import ConsistencyError

from covid_triage.io import load_dicoms, save_image, save_covid_label, save_lungs_mask


def load_nii_arr(file):
    arr = nibabel.as_closest_canonical(nibabel.load(file)).get_fdata()
    arr = np.swapaxes(arr[::-1, ::-1, ::-1], 0, 1)
    return arr


def main(src, dst):
    src = Path(src)
    dst = Path(dst)
    for patient_path in tqdm(list((src / 'NSCLC-Radiomics').glob('*'))):
        ct_path = min(patient_path.glob('*/*'), key=lambda path: datetime.strptime(path.name, '%m-%d-%Y'))
            
        series = load_dicoms(ct_path)
        series_uid = get_common_tag(series, 'SeriesInstanceUID')
        if get_slices_plane(series) != Plane.Axial:
            raise ValueError('Series is not axial.')
        series = order_series(series)
        image = stack_images(series, -1).astype(np.int16)
        try:
            spacing = get_voxel_spacing(series)
        except ConsistencyError as e:
            print(f'Series {series_uid} has inconsistent spacing.')
            continue
        save_image(dst, series_uid, image, spacing)

        files = list((src / 'Thoracic_Cavities' / patient_path.name).glob('*.nii.gz'))
        if files:
            lungs_mask = load_nii_arr(files[0]).astype(bool)
            assert lungs_mask.shape == image.shape, patient_path
            save_lungs_mask(dst, series_uid, lungs_mask)

        save_covid_label(dst, series_uid, covid_label=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input')
    parser.add_argument('-o', '--output')
    args = parser.parse_args()

    main(args.input, args.output)
