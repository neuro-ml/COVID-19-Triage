import argparse
from functools import partial
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, recall_score, precision_score, accuracy_score
import torch
from torch import nn
import torch.nn.functional as F

from dpipe.dataset.wrappers import merge, cache_methods
from dpipe.batch_iter import Infinite, load_by_random_id, unpack_args, apply_at, \
    random_apply, multiply, combine_pad, sample_args
from dpipe.im.shape_utils import prepend_dims
from dpipe import layers
from dpipe.torch import train_step, masked_loss, inference_step, save_model_state
from dpipe.predict import add_extract_dims
from dpipe.train.validator import compute_metrics
from dpipe.train import train, Schedule, TBLogger, TQDM, TimeProfiler

from covid_triage.dataset import Dataset, RescaleToPixelSpacing, CropToLungs, NormalizeIntensities
from covid_triage.utils import invert_stratification_mapping
from covid_triage.batch_iter import get_sampling_weights, sample_mask_from_dict, drop_slices
from covid_triage.model import get_covid_multitask_spatial_model
from covid_triage.predictor import multi_inference_step
from covid_triage.checkpoint import CheckpointWithBestMetrics

# preprocessing hyperparameters
PIXEL_SPACING = (2, 2)  # images are rescaled to the fixed pixel spacing
WINDOW = (-1000, 300)  # image intensities are clipped to the fixed window and rescaled to (0, 1)

# cross validation hyperparameters
RANDOM_SEED = 42
VAL_SIZE = 30

# training hyperparameters
NUM_SLICES = 32  # we leave each k-th axial slice, such that remaining series has length NUM_SLICES
BATCH_SIZE = 5
NUM_EPOCHS = 100
NUM_BATCHES_PER_EPOCH = 300
CLS_WEIGHT = .1  # BCE for classifiction task is multitplied by this weight before adding to BCE for segmentation task
LEARNING_RATE = Schedule(initial=3e-4, epoch2value_multiplier={100: .3, 200: .3})

DEVICE = 'cuda'


def main(mosmed_root, medseg_root, nsclc_root, dst):
    dst = Path(dst)

    # dataset
    mosmed = Dataset(root=mosmed_root)
    medseg = Dataset(root=medseg_root)
    nsclc = Dataset(root=nsclc_root)
    raw_dataset = merge(mosmed, medseg, nsclc)
    dataset = cache_methods(NormalizeIntensities(CropToLungs(RescaleToPixelSpacing(raw_dataset, PIXEL_SPACING)), WINDOW))

    # cross validation design
    mosmed_positive_ids = sorted(filter(dataset.load_covid_label, mosmed.ids)) 
    nsclc_ids_with_lungs_mask = sorted(filter(nsclc.has_lungs_mask, nsclc.ids))
    negative_ids = sorted(set(mosmed.ids) - set(mosmed_positive_ids)) + nsclc_ids_with_lungs_mask
    positive_ids_with_mask = sorted(filter(dataset.has_covid_masks, mosmed_positive_ids)) + medseg.ids
    positive_ids_without_mask = sorted(set(mosmed_positive_ids) - set(positive_ids_with_mask))
    ids_without_positive_mask = negative_ids + positive_ids_without_mask

    stratification_label_to_ids = {
        'negative': negative_ids,
        'positive_without_mask': positive_ids_without_mask,
        'positive_with_mask': positive_ids_with_mask
    }
    id_to_stratification_label = invert_stratification_mapping(stratification_label_to_ids)

    train_ids, val_ids = train_test_split(ids_without_positive_mask, test_size=VAL_SIZE, random_state=RANDOM_SEED, 
        stratify=[id_to_stratification_label[id_] for id_ in ids_without_positive_mask])
    train_ids += positive_ids_with_mask
    
    # batch iterator
    label_to_sampling_weight = {
        # used only to penalize classification head
        'negative': .25,
        'positive_without_mask': .25,
        # used only to penalize segmentation head
        'positive_with_mask': .5
    }
    sampling_weights = get_sampling_weights(train_ids, id_to_stratification_label, label_to_sampling_weight)
    batch_iter = Infinite(
        load_by_random_id(dataset.load_image, dataset.load_covid_masks, dataset.load_covid_label, 
                          ids=train_ids, weights=sampling_weights),
        unpack_args(lambda image, masks, label: (
            image, sample_mask_from_dict(masks, shape=image.shape), label, 
        )),
        unpack_args(drop_slices, n_slices_left=NUM_SLICES, n_scalars=1),
        multiply(prepend_dims),
        multiply(np.float32),
        batch_size=BATCH_SIZE, 
        batches_per_epoch=NUM_BATCHES_PER_EPOCH,
        buffer_size=BATCH_SIZE,
        combiner=combine_pad
    )

    # neural network architecture
    model = get_covid_multitask_spatial_model().to(DEVICE)

    # optimizer 
    optimizer = torch.optim.Adam(model.parameters())


    # criterion
    def criterion(output, covid, is_covid):
        """
        output: (sgm_logits, cls_logits); sgm_logits: (N, 1, H, W, D), cls_logits: (N, 1)
        covid: (N, 1, H, W, D), can contain Nans
        is_covid: (N, 1)
        """
        sgm_logits, cls_logits = output
        sgm_loss_mask = ~torch.isnan(covid) & is_covid[..., None, None, None].bool()
        sgm_bce = masked_loss(sgm_loss_mask, F.binary_cross_entropy_with_logits, sgm_logits, covid)
        cls_bce = F.binary_cross_entropy_with_logits(cls_logits, is_covid)
        return {
            'loss': CLS_WEIGHT * cls_bce + sgm_bce, 
            'cls_bce': cls_bce, 
            'sgm_bce': sgm_bce,
        }


    # predict 
    @add_extract_dims(2, sequence=True)
    def predict(image):
        """
        Input: an image of size (H, W, D) which is normalized, zoomed to spacing and cropped to lungs.
        Output: the COVID-19 lesions probability map and the probability of being COVID-19 positive.
        """
        return multi_inference_step(image, architecture=model, actvations=torch.sigmoid)


    # validation metrics
    metrics = {
        'roc_auc': roc_auc_score,
        'recall': recall_score,
        'precision': precision_score,
        'accuracy': accuracy_score,
    }


    def validate():
        return compute_metrics(predict, dataset.load_image, dataset.load_covid_label, val_ids, metrics)


    def are_better(current_metrics, best_metrics):
        return current_metrics['roc_auc'] >= best_metrics['roc_auc']


    # train
    logger = TBLogger(dst / 'logs')
    train(
        train_step=train_step,
        batch_iter=batch_iter,
        n_epochs=NUM_EPOCHS,
        logger=logger,
        checkpoints=CheckpointWithBestMetrics(dst / 'checkpoints', [model, optimizer], are_better),
        time=TimeProfiler(logger.logger),
        validate=validate,
        architecture=model,
        optimizer=optimizer,
        criterion=criterion,
        lr=LEARNING_RATE,
        n_targets=2,
        loss_key='loss',
        tqdm=TQDM(),
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mosmed')
    parser.add_argument('--medseg')
    parser.add_argument('--nsclc')
    parser.add_argument('-o', '--output')
    args = parser.parse_args()

    main(args.mosmed, args.medseg, args.nsclc, args.output)
