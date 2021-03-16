import numpy as np
import torch
from dpipe.torch import inference_step, sequence_to_var, to_np
from dpipe.itertools import zip_equal, collect
from dpipe.im import zoom, zoom_to_shape, crop_to_box, restore_crop
from dpipe.im.slices import iterate_axis
from dpipe.im.shape_utils import prepend_dims, extract_dims
from dpipe.im.box import mask2bounding_box
from dpipe.im.utils import identity
from dpipe.predict import add_extract_dims
from .processing import normalize_intensities, zoom_to_shape_within_box


def _preprocess(image, spacing, new_pixel_spacing, window):
    scale_factor = spacing[:2] / np.broadcast_to(new_pixel_spacing, 2)
    image = zoom(image, scale_factor, axis=(0, 1), fill_value=np.min)
    return normalize_intensities(image, *window)


def _predict_lungs_sgm_probas(image, lungs_sgm_model):
    slice_segmentations = [
        inference_step(slc, architecture=lungs_sgm_model, activation=torch.sigmoid)
        for slc in iterate_axis(prepend_dims(image, 2), -1)
    ]
    return extract_dims(np.stack(slice_segmentations, -1), 2)


def get_lungs_mask_predictor(lungs_sgm_model, pixel_spacing, window, sgm_probas_threshold):
    def predict(image, spacing):
        original_shape = image.shape
        image = _preprocess(image, spacing, pixel_spacing, window)
        sgm_probas = _predict_lungs_sgm_probas(image, lungs_sgm_model)

        # postprocessing
        box = mask2bounding_box(sgm_probas >= sgm_probas_threshold)
        sgm_probas = zoom_to_shape_within_box(sgm_probas, original_shape, box)
        
        return sgm_probas >= sgm_probas_threshold

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


def get_covid_multitask_predictor(lungs_sgm_model, covid_mutitask_model, pixel_spacing, window, 
                                  lungs_sgm_threshold, covid_sgm_theshold):
    def predict(image, spacing):
        original_shape = image.shape

        # common preprocessing for lungs segmentation and covid prediction
        image = _preprocess(image, spacing, pixel_spacing, window)

        # lungs segmentation
        lungs_sgm_probas = _predict_lungs_sgm_probas(image, lungs_sgm_model)
        lungs_mask = lungs_sgm_probas >= lungs_sgm_threshold
        if not lungs_mask.any():
            raise RuntimeError('Predicted lungs mask is empty.')
        
        # crop to lungs box, as a final preprocessing step before covid prediction
        lungs_box = mask2bounding_box(lungs_mask)
        image = crop_to_box(image, lungs_box)

        # covid prediction
        feature_maps = [
            inference_step(slc, architecture=covid_mutitask_model.backbone, activation=identity)
            for slc in iterate_axis(prepend_dims(image, 2), -1)
        ]
        fm = np.stack(feature_maps, -1)

        slice_segmentations = [
            inference_step(slc, architecture=covid_mutitask_model.sgm_head, activation=torch.sigmoid)
            for slc in feature_maps
        ]
        covid_sgm_probas = extract_dims(np.stack(slice_segmentations, -1), 2)
        
        covid_proba = inference_step(fm, architecture=covid_mutitask_model.cls_head, activation=torch.sigmoid)
        covid_proba = extract_dims(covid_proba, 2)

        # postprocessing
        covid_sgm_probas = restore_crop(covid_sgm_probas, lungs_box, lungs_mask.shape)
        covid_sgm_probas = zoom_to_shape_within_box(covid_sgm_probas, original_shape, lungs_box)
        covid_mask = covid_sgm_probas >= covid_sgm_theshold

        return covid_mask, covid_proba

    return predict


def get_resnet_predictor(lungs_sgm_model, resnet, pixel_spacing, window, lungs_sgm_threshold):
    def predict(image, spacing):
        original_shape = image.shape

        # common preprocessing for lungs segmentation and covid prediction
        image = _preprocess(image, spacing, pixel_spacing, window)

        # lungs segmentation
        lungs_sgm_probas = _predict_lungs_sgm_probas(image, lungs_sgm_model)
        lungs_mask = lungs_sgm_probas >= lungs_sgm_threshold
        if not lungs_mask.any():
            raise RuntimeError('Predicted lungs mask is empty.')
        
        # crop to lungs box, as a final preprocessing step before covid prediction
        lungs_box = mask2bounding_box(lungs_mask)
        image = crop_to_box(image, lungs_box)

        covid_proba = extract_dims(inference_step(
            prepend_dims(image, 2), architecture=resnet, activation=torch.sigmoid), 2)
        
        return covid_proba

    return predict