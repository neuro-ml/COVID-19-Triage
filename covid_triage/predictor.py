import numpy as np
import torch
from dpipe.torch import inference_step, sequence_to_var, to_np
from dpipe.itertools import zip_equal, collect
from dpipe.im import zoom, zoom_to_shape
from dpipe.im.slices import iterate_axis
from dpipe.im.shape_utils import prepend_dims, extract_dims
from dpipe.im.box import mask2bounding_box
from dpipe.predict import add_extract_dims
from .processing import normalize_intensities


def get_lungs_mask_predictor(lungs_sgm_model, pixel_spacing, window):
    def predict(image, spacing):
        original_shape = image.shape

        # preprocessing
        scale_factor = np.broadcast_to(pixel_spacing, 2) / spacing[:2]
        image = zoom(image, scale_factor, axis=(0, 1), fill_value=np.min)
        image = normalize_intensities(image, *window)

        # apply network
        slice_predictions = [
            inference_step(slc, architecture=lungs_sgm_model, activation=torch.sigmoid)
            for slc in iterate_axis(prepend_dims(image, 2), -1)
        ]
        sgm_probas = extract_dims(np.stack(slice_predictions, -1), 2)

        # postprocessing
        sgm_probas = zoom_to_shape(sgm_probas, original_shape)
        
        return sgm_probas >= .5

    return predict


@collect
def multi_inference_step(*inputs, architecture, activations):
    architecture.eval()
    with torch.no_grad():
        preds = architecture(*sequence_to_var(*inputs, device=architecture))
        if callable(activations):
            activations = [activations] * len(preds)
        for x, act in zip_equal(preds, activations):
            yield to_np(act(x))
