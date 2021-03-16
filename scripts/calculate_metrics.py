import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

from dpipe.io import load
from dpipe.im.metrics import dice_score

from covid_triage.dataset import Dataset
from covid_triage.utils import get_lungs_and_lesions_volumes


def main(test_root, predictions_root):
    predictions_root = Path(predictions_root)
    dataset = Dataset(test_root)
    labels = {id_: dataset.load_covid_label(id_) for id_ in dataset.ids}
    try:
        probas = {id_: load(predictions_root / id_ / 'covid_proba.json') for id_ in dataset.ids}

        def roc_auc(ids):
            y_true = [labels[id_] for id_ in ids]
            y_score = [probas[id_] for id_ in ids]
            return roc_auc_score(y_true, y_score)

        print(f'ROC AUC COVID-19 vs. All: {roc_auc(dataset.ids):.2f}')
        print(f'ROC AUC COVID-19 vs. Bacterial Pneumonia: {roc_auc(dataset.covid_ids + dataset.bacterial_pneumonia_ids):.2f}.')
        print(f'ROC AUC COVID-19 vs. Lung Nodules: {roc_auc(dataset.covid_ids + dataset.nodules_ids):.2f}.')
        print(f'ROC AUC COVID-19 vs. Normal: {roc_auc(dataset.covid_ids + dataset.normal_ids):.2f}.')

    except FileNotFoundError:
        print('Predicted covid proba not found.')

    try:   
        print('Calculating segmentation metrics...')
        fractions_gt = []
        fractions_pred = []
        dices = []
        for id_ in tqdm(dataset.covid_ids):
            lungs_mask = dataset.load_lungs_mask(id_)
            masks_gt = dataset.load_covid_masks(id_)
            mask_pred = load(predictions_root / id_ / 'lesions_mask.npy')
            spacing = dataset.load_spacing(id_)

            avg_dice = np.mean([dice_score(mask_gt, mask_pred) for mask_gt in masks_gt.values()])
            dices.append(avg_dice)

            fraction_gt = []
            for mask_gt in masks_gt.values():
                lungs_volumes, lesions_volumes = get_lungs_and_lesions_volumes(lungs_mask, mask_gt, spacing)
                fraction_gt.append(max(lesions_volumes / lungs_volumes))
            fractions_gt.append(np.mean(fraction_gt))

            lungs_volumes, lesions_volumes = get_lungs_and_lesions_volumes(lungs_mask, mask_pred, spacing)
            fractions_pred.append(max(lesions_volumes / lungs_volumes))

        r, p = spearmanr(fractions_gt, fractions_pred)

        print(f'Dice: {np.mean(dices):.2f}.')
        print(f"Spearmans' rho: {r:.2f} (p-value {p:.4f}).")

    except FileNotFoundError:
        print('Predicted lesions mask not found.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test')
    parser.add_argument('--pred')
    args = parser.parse_args()

    main(args.test, args.pred)
