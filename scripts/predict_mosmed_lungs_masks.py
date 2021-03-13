import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm

from dpipe.torch import load_model_state

from covid_triage.dataset import Dataset, RescaleToPixelSpacing, NormalizeIntensities
from covid_triage.model import get_lungs_sgm_model
from covid_triage.predictor import get_lungs_mask_predictor
from covid_triage.io import save_lungs_mask
from covid_triage.const import PIXEL_SPACING, WINDOW

DEVICE = 'cuda'


def main(lungs_sgm_model_path, mosmed_root):
    mosmed_root = Path(mosmed_root)
    mosmed = Dataset(root=mosmed_root)

    model = get_lungs_sgm_model().to(DEVICE)
    load_model_state(model, lungs_sgm_model_path)
    predict = get_lungs_mask_predictor(model, PIXEL_SPACING, WINDOW)

    for id_ in tqdm(mosmed.ids):
        image = mosmed.load_image(id_)
        spacing = mosmed.load_spacing(id_)
        lungs_mask = predict(image, spacing)

        save_lungs_mask(mosmed_root, id_, lungs_mask)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model')
    parser.add_argument('--mosmed')
    args = parser.parse_args()

    main(args.model, args.mosmed)
    