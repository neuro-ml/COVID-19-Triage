import argparse
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F

from dpipe.dataset.wrappers import cache_methods
from dpipe.batch_iter import Infinite, load_by_random_id, unpack_args, apply_at, \
    random_apply, multiply, combine_pad, sample_args
from dpipe.im.shape_utils import prepend_dims
from dpipe.im.slices import iterate_axis
from dpipe.torch import train_step, save_model_state, inference_step
from dpipe.predict import add_extract_dims
from dpipe.im.metrics import dice_score, convert_to_aggregated
from dpipe.train.validator import compute_metrics
from dpipe.train import train, Schedule, TBLogger, TQDM, TimeProfiler

from covid_triage.dataset import Dataset, RescaleToPixelSpacing, NormalizeIntensities
from covid_triage.batch_iter import sample_slice
from covid_triage.checkpoint import CheckpointWithBestMetrics
from covid_triage.model import get_lungs_sgm_model
from covid_triage.const import PIXEL_SPACING, WINDOW, LUNGS_SGM_THRESHOLD

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
    dataset = cache_methods(NormalizeIntensities(RescaleToPixelSpacing(nsclc, PIXEL_SPACING), WINDOW))

    # cross validation design
    ids = sorted(filter(dataset.has_lungs_mask, dataset.ids))
    train_ids, val_ids = train_test_split(ids, test_size=VAL_SIZE, random_state=RANDOM_SEED)
    
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

    # torch model
    model = get_lungs_sgm_model().to(DEVICE)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters())

    # predict 
    @add_extract_dims(2)
    def predict(image):
        slice_predictions = [
            inference_step(slc, architecture=model, activation=torch.sigmoid)
            for slc in iterate_axis(image, -1)
        ]
        return np.stack(slice_predictions, -1) >= LUNGS_SGM_THRESHOLD

    # validation metrics
    metrics = convert_to_aggregated({'dice': dice_score})


    def validate():
        return compute_metrics(predict, dataset.load_image, dataset.load_lungs_mask, val_ids, metrics)


    def are_better(current_metrics, best_metrics):
        return current_metrics['dice'] >= best_metrics['dice']

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
    parser.add_argument('--nsclc')
    parser.add_argument('-o', '--output')
    args = parser.parse_args()

    main(args.nsclc, args.output)
