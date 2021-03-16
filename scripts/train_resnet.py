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
from dpipe.itertools import dmap
from dpipe.batch_iter import Infinite, load_by_random_id, unpack_args, apply_at, \
    random_apply, multiply, combine_pad, sample_args
from dpipe.im.shape_utils import prepend_dims
from dpipe import layers
from dpipe.torch import train_step, inference_step
from dpipe.predict import add_extract_dims
from dpipe.train.validator import compute_metrics
from dpipe.train import train, Schedule, TBLogger, TQDM, TimeProfiler

from covid_triage.dataset import Dataset, RescaleToPixelSpacing, CropToLungs, NormalizeIntensities
from covid_triage.utils import invert_stratification_mapping
from covid_triage.batch_iter import get_sampling_weights, drop_slices
from covid_triage.processing import crop_to_max_shape_xy
from covid_triage.model import get_resnet
from covid_triage.checkpoint import CheckpointWithBestMetrics
from covid_triage.const import PIXEL_SPACING, WINDOW

# cross validation hyperparameters
RANDOM_SEED = 42
VAL_SIZE = 30

# training hyperparameters
NUM_SLICES = 32  # we leave each k-th axial slice, such that remaining series has length NUM_SLICES
MAX_SHAPE_XY = (128, 160)  # we crop image in Oxy plane, if its size is too large, in order to fit the GPU memory
BATCH_SIZE = 2
NUM_EPOCHS = 100
NUM_BATCHES_PER_EPOCH = 300
LEARNING_RATE = Schedule(initial=3e-4, epoch2value_multiplier={80: .3})

DEVICE = 'cuda'


def main(mosmed_root, nsclc_root, dst):
    dst = Path(dst)

    # dataset
    mosmed = Dataset(root=mosmed_root)
    nsclc = Dataset(root=nsclc_root)
    raw_dataset = merge(mosmed, nsclc)
    dataset = cache_methods(NormalizeIntensities(CropToLungs(RescaleToPixelSpacing(raw_dataset, PIXEL_SPACING)), WINDOW))

    # cross validation design
    mosmed_positive_ids = sorted(filter(dataset.load_covid_label, mosmed.ids)) 
    nsclc_ids_with_lungs_mask = sorted(filter(nsclc.has_lungs_mask, nsclc.ids))
    negative_ids = sorted(set(mosmed.ids) - set(mosmed_positive_ids)) + nsclc_ids_with_lungs_mask
    mosmed_positive_ids_with_mask = sorted(filter(dataset.has_covid_masks, mosmed_positive_ids))
    mosmed_positive_ids_without_mask = sorted(set(mosmed_positive_ids) - set(mosmed_positive_ids_with_mask))
    ids_without_positive_mask = negative_ids + mosmed_positive_ids_without_mask

    stratification_label_to_ids = {
        'negative': negative_ids,
        'positive': mosmed_positive_ids_without_mask,
    }
    id_to_stratification_label = invert_stratification_mapping(stratification_label_to_ids)

    train_ids, val_ids = train_test_split(ids_without_positive_mask, test_size=VAL_SIZE, random_state=RANDOM_SEED, 
        stratify=[id_to_stratification_label[id_] for id_ in ids_without_positive_mask])
    
    # batch iterator
    label_to_sampling_weight = {
        'negative': .5,
        'positive': .5
    }
    sampling_weights = get_sampling_weights(train_ids, id_to_stratification_label, label_to_sampling_weight)
    batch_iter = Infinite(
        load_by_random_id(dataset.load_image, dataset.load_covid_label, ids=train_ids, weights=sampling_weights),
        apply_at(0, crop_to_max_shape_xy, max_shape_xy=MAX_SHAPE_XY),
        unpack_args(drop_slices, n_slices_left=NUM_SLICES, n_scalars=1),
        multiply(prepend_dims),
        multiply(np.float32),
        batch_size=BATCH_SIZE,
        batches_per_epoch=NUM_BATCHES_PER_EPOCH,
        buffer_size=BATCH_SIZE,
        combiner=combine_pad
    )

    # neural network architecture
    model = get_resnet().to(DEVICE)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters())


    # predict 
    @add_extract_dims(2)
    def predict(image):
        return inference_step(image, architecture=model, activation=torch.sigmoid)


    # validation metrics
    def cls_metric_decorator(metric, threshold):
        def wrapper(targets, predictions):
            return metric(targets, [p >= threshold for p in predictions])
        return wrapper


    metrics = {
        'roc_auc': roc_auc_score,
        'recall': cls_metric_decorator(recall_score, threshold=.5),
        'precision': cls_metric_decorator(precision_score, threshold=.5),
        'accuracy': cls_metric_decorator(accuracy_score, threshold=.5),
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
        criterion=F.binary_cross_entropy_with_logits,
        lr=LEARNING_RATE,
        n_targets=1,
        tqdm=TQDM(loss=False),
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mosmed')
    parser.add_argument('--nsclc')
    parser.add_argument('-o', '--output')
    args = parser.parse_args()

    main(args.mosmed, args.nsclc, args.output)
