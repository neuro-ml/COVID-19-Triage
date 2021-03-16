import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm

from dpipe.torch import load_model_state

from covid_triage.dataset import Dataset, RescaleToPixelSpacing, NormalizeIntensities
from covid_triage.model import get_lungs_sgm_model
from covid_triage.predictor import get_lungs_mask_predictor
from covid_triage.io import save_lungs_mask
from covid_triage.const import PIXEL_SPACING, WINDOW, LUNGS_SGM_THRESHOLD

DEVICE = 'cuda'


def main(lungs_sgm_model_path, dataset_root):
    dataset = Dataset(root=dataset_root)

    model = get_lungs_sgm_model().to(DEVICE)
    load_model_state(model, lungs_sgm_model_path)
    predict = get_lungs_mask_predictor(model, PIXEL_SPACING, WINDOW, LUNGS_SGM_THRESHOLD)

    for id_ in tqdm(dataset.ids):
        image = dataset.load_image(id_)
        spacing = dataset.load_spacing(id_)
        lungs_mask = predict(image, spacing)

        save_lungs_mask(dataset_root, id_, lungs_mask)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model')
    parser.add_argument('--dataset')
    args = parser.parse_args()

    main(args.model, args.dataset)
    