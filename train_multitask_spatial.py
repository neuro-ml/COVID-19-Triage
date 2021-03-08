import argparse
from functools import partial
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, recall_score, precision_score, accuracy_score
import torch
from torch import nn
import torch.nn.functional as F

from dpipe.dataset.wrappers import merge, apply
from dpipe.batch_iter import Infinite, load_by_random_id, unpack_args, apply_at, \
    random_apply, multiply, combine_pad, sample_args
from dpipe.im.shape_utils import prepend_dims
from dpipe import layers
from dpipe.torch import train_step, masked_loss, inference_step
from dpipe.train.validator import compute_metrics
from dpipe.train import train, Schedule, TBLogger, TQDM, TimeProfiler

from covid_triage.dataset import Dataset, RescaleToPixelSpacing, CropToLungs, NormalizeIntensities
from covid_triage.utils import invert_stratification_mapping
from covid_triage.batch_iter import get_sampling_weights, sample_mask_from_dict, drop_slices
from covid_triage.architecture import SliceWise
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mosmed')
    parser.add_argument('--medseg')
    parser.add_argument('--nsclc')
    args = parser.parse_args()
    
    # dataset
    mosmed = Dataset(root=parser.mosmed)
    medseg = Dataset(root=parser.medseg)
    nsclc = Dataset(root=parser.nsclc)
    raw_dataset = merge(mosmed, medseg, nsclc)
    dataset = NormalizeIntensities(CropToLungs(RescaleToPixelSpacing(raw_dataset, PIXEL_SPACING)), WINDOW)

    # cross validation design
    mosmed_positive_ids = sorted(filter(dataset.load_covid_label, mosmed.ids)) 
    negative_ids = sorted(set(mosmed.ids) - mosmed_positive_ids) + nsclc.ids
    positive_ids_with_mask = sorted(filter(dataset.has_covid_masks, mosmed_positive_ids)) + medseg.ids
    positive_ids_without_mask = sorted(set(mosmed_positive_ids) - set(positive_ids_with_mask))
    ids_without_positive_mask = negative_ids + positive_ids_without_mask

    stratification_label_to_ids = {
        'negative': negative_ids,
        'positive_without_mask': positive_ids_without_mask,
        'positive_with_mask': positive_ids_with_mask
    }
    id_to_stratifiction_label = invert_stratification_mapping(stratification_label_to_ids)

    train_ids, val_ids = train_test_split(ids_without_positive_mask, test_size=VAL_SIZE, random_state=RANDOM_SEED, 
        stratify=[id_to_stratifiction_label[id_] for id_ in ids_without_positive_mask])
    train_ids += positive_ids_with_mask
    
    # batch iterator
    label_to_sampling_weight = {
        # used only to penalize classification head
        'negative': .25,
        'mosmed_positive_without_mask': .25,
        # used only to penalize segmentation head
        'positive_with_mask': .5
    }
    sampling_weights = get_sampling_weights(train_ids, id_to_stratifiction_label, label_to_sampling_weight)
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
    backbone = SliceWise(nn.Sequential(
        nn.Conv2d(1, 8, kernel_size=3, padding=1),
        layers.FPN(
            layer=layers.ResBlock2d,
            downsample=nn.MaxPool2d(2, ceil_mode=True),
            upsample=nn.Identity,
            merge=lambda left, down: torch.add(*layers.interpolate_to_left(left, down, 'bilinear')),
            structure=[
                [[8, 8, 8], [8, 8]],
                [[8, 16, 16], [16, 16, 8]],
                [[16, 32, 32], [32, 32, 16]],
                [[32, 64, 64], [64, 64, 32]],
                [[64, 128, 128], [128, 128, 64]],
                [[128, 256, 256], [256, 256, 128]],
                [256, 512, 256]
            ],
            kernel_size=3,
            padding=1
        ),
    ))
    sgm_head = SliceWise(nn.Sequential(
        layers.ResBlock2d(8, 8, kernel_size=3, padding=1),
        layers.PreActivation2d(8, 1, kernel_size=1, bias=False),  # shape (N, 1, H, W, D)
    ))
    cls_head = nn.Sequential(
        layers.PyramidPooling(partial(F.max_pool3d, ceil_mode=True), levels=4),
        nn.Dropout(),
        nn.Linear(layers.PyramidPooling.get_multiplier(levels=4, ndim=3) * 8, 1024),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(1024, 1, bias=False),  # shape (N, 1)
    )
    architecture = nn.Sequential(backbone, layers.Split(sgm_head, cls_head))

    # optimizer 
    optimizer = torch.optim.Adam(architecture.parameters())


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
        fm = inference_step(image, architecture=backbone)
        sgm_probas = inference_step(fm, architecture=sgm_head, activation=torch.sigmoid)
        cls_proba = inference_step(fm, architecture=cls_head, activation=torch.sigmoid)
        return sgm_probas, cls_proba


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
    logger = TBLogger('logs')
    train(
        train_step=train_step,
        batch_iter=batch_iter,
        n_epochs=NUM_EPOCHS,
        logger=logger,
        checkpoints=CheckpointWithBestMetrics('checkpoints', [architecture, optimizer], are_better),
        time=TimeProfiler(logger.logger),
        validate=validate,
        architecture=architecture,
        optimizer=optimizer,
        criterion=criterion,
        lr=LEARNING_RATE,
        n_targets=2,
        loss_key='loss',
        tqdm=TQDM(),
    )
