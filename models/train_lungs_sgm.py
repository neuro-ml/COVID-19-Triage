import argparse
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import torch
from torch import nn
import torch.nn.functional as F

from dpipe.batch_iter import Infinite, load_by_random_id, unpack_args, apply_at, \
    random_apply, multiply, combine_pad, sample_args
from dpipe.im.shape_utils import prepend_dims
from dpipe.im.slices import iterate_axis
from dpipe import layers
from dpipe.torch import train_step, inference_step, save_model_state
from dpipe.im.metrics import dice_score
from dpipe.train.validator import compute_metrics
from dpipe.train import train, Schedule, TBLogger, TQDM, TimeProfiler

from covid_triage.dataset import Dataset, RescaleToPixelSpacing, NormalizeIntensities
from covid_triage.batch_iter import sample_slice
from covid_triage.checkpoint import CheckpointWithBestMetrics

# preprocessing hyperparameters
PIXEL_SPACING = (2, 2)  # images are rescaled to the fixed pixel spacing
WINDOW = (-1000, 300)  # image intensities are clipped to the fixed window and rescaled to (0, 1)

# cross validation hyperparameters
RANDOM_SEED = 42
VAL_SIZE = 30

# training hyperparameters
BATCH_SIZE = 30
NUM_EPOCHS = 100
NUM_BATCHES_PER_EPOCH = 160
LEARNING_RATE = Schedule(initial=1e-3, epoch2value_multiplier={50: .1})

DEVICE = 'cuda'


def main(nsclc_root, dst):
    dst = Path(dst)

    # dataset
    nsclc = Dataset(root=nsclc_root)
    dataset = NormalizeIntensities(RescaleToPixelSpacing(nsclc, PIXEL_SPACING), WINDOW)

    # cross validation design
    train_ids, val_ids = train_test_split(dataset.ids, test_size=VAL_SIZE, random_state=RANDOM_SEED)
    
    # batch iterator
    batch_iter = Infinite(
        load_by_random_id(dataset.load_image, dataset.load_lungs_mask, ids=train_ids),
        unpack_args(sample_slice),
        multiply(prepend_dims),
        multiply(np.float32),
        batch_size=BATCH_SIZE, 
        batches_per_epoch=NUM_BATCHES_PER_EPOCH,
        buffer_size=BATCH_SIZE,
        combiner=combine_pad
    )

    # neural network architecture
    architecture = nn.Sequential(
        nn.Conv2d(1, 8, kernel_size=3, padding=1),
        layers.FPN(
            layer=layers.ResBlock2d,
            downsample=nn.MaxPool2d(2, ceil_mode=True),
            upsample=nn.Identity,
            merge=lambda left, down: torch.add(*layers.interpolate_to_left(left, down, 'bilinear')),
            structure=[
                [[8, 8, 8], [8, 8, 8]],
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
        layers.PreActivation2d(8, 1, kernel_size=1, bias=False)
    )

    # optimizer 
    optimizer = torch.optim.Adam(architecture.parameters())

    # predict 
    @add_extract_dims(2, sequence=True)
    def predict(image):
        return np.stack([
            inference_step(slc, architecture=architecture, activation=torch.sigmoid) 
            for slc in iterate_axis(image, -1)
        ], -1)


    # validation metrics
    metrics = {'dice': dice_score}


    def validate():
        return compute_metrics(predict, dataset.load_image, dataset.load_lungs_mask, val_ids, metrics)


    def are_better(current_metrics, best_metrics):
        return current_metrics['dice'] >= best_metrics['dice']


    # train
    logger = TBLogger(dst / 'logs')
    architecture.to(DEVICE)
    train(
        train_step=train_step,
        batch_iter=batch_iter,
        n_epochs=NUM_EPOCHS,
        logger=logger,
        checkpoints=CheckpointWithBestMetrics(dst / 'checkpoints', [architecture, optimizer], are_better),
        time=TimeProfiler(logger.logger),
        validate=validate,
        architecture=architecture,
        optimizer=optimizer,
        criterion=F.binary_cross_entropy_with_logits,
        lr=LEARNING_RATE,
        n_targets=1,
        loss_key='loss',
        tqdm=TQDM(),
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nsclc')
    parser.add_argument('-o', '--output')
    args = parser.parse_args()

    main(args.nsclc, args.output)
