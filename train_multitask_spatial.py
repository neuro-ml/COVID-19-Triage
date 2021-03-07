import argparse
import numpy as np
from sklearn.model_selection import train_test_split

from dpipe.train import train, Schedule, TBLogger
from dpipe.torch import train_step
from dpipe.dataset.wrappers import merge, apply
from dpipe.batch_iter import Infinite, load_by_random_id, unpack_args, apply_at, \
    random_apply, multiply, combine_pad, sample_args
from dpipe.im.shape_utils import prepend_dims

from covid_triage.dataset import Dataset, RescaleToPixelSpacing, CropToLungs, NormalizeIntensities
from covid_triage.utils import invert_stratification_mapping
from covid_triage.batch_iter import get_sampling_weights, sample_mask_from_dict, drop_slices
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

    # train(
    #     train_step=train_step,
    #     batch_iter=batch_iter,
    #     n_epochs=NUM_EPOCHS,
    #     logger=TBLogger('logs'),
    #     lr=LEARNING_RATE
    # )
