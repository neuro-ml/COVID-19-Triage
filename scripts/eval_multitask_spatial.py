import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm

from dpipe.torch import load_model_state
from dpipe.io import save

from covid_triage.dataset import Dataset, RescaleToPixelSpacing, NormalizeIntensities
from covid_triage.model import get_lungs_sgm_model, MultitaskSpatial
from covid_triage.predictor import get_covid_multitask_predictor
from covid_triage.const import PIXEL_SPACING, WINDOW, LUNGS_SGM_THRESHOLD, COVID_SGM_THRESHOLD

DEVICE = 'cuda'


def main(lungs_sgm_model_path, multitask_spatial_model_path, test_root, dst):
    dst = Path(dst)
    test_root = Path(test_root)
    dataset = Dataset(root=test_root)

    lungs_model = get_lungs_sgm_model().to(DEVICE)
    load_model_state(lungs_model, lungs_sgm_model_path)

    covid_model = MultitaskSpatial().to(DEVICE)
    load_model_state(covid_model, multitask_spatial_model_path)

    predict = get_covid_multitask_predictor(lungs_model, covid_model, PIXEL_SPACING, WINDOW, 
        LUNGS_SGM_THRESHOLD, COVID_SGM_THRESHOLD)

    for id_ in tqdm(dataset.ids):
        image = dataset.load_image(id_)
        spacing = dataset.load_spacing(id_)
        covid_mask, covid_proba = predict(image, spacing)

        (dst / id_).mkdir(parents=True, exist_ok=True)
        save(covid_mask, dst / id_ / 'lesions_mask.npy')
        save(covid_proba, dst / id_ / 'covid_proba.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lungs_model')
    parser.add_argument('--multitask_spatial')
    parser.add_argument('--test')
    parser.add_argument('-o', '--output')
    args = parser.parse_args()

    main(args.lungs_model, args.multitask_spatial, args.test, args.output)
    