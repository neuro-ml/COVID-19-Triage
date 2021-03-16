from pathlib import Path
import numpy as np

from dpipe.io import save, load
from dpipe.itertools import dmap
from dpipe.dataset.wrappers import Proxy
from dpipe.im.utils import composition
from dpipe.im import crop_to_box, zoom
from dpipe.im.box import mask2bounding_box

from .processing import normalize_intensities


class Dataset:
    def __init__(self, root):
        root = Path(root)
        self.root = root
        self.ids = sorted(path.name for path in root.glob('images/*'))
        for subset in self.root.glob('subsets/*'):
            subset_ids = load(subset / 'ids.json')
            setattr(self, f'{subset.name}_ids', subset_ids)

    def load_image(self, id_):
        return load(self.root / 'images' / id_ / 'image.npy')

    def load_spacing(self, id_):
        return load(self.root / 'images' / id_ / 'spacing.json')

    def load_covid_masks(self, id_):
        """
        Returns dictionary that maps rater to covid mask. 
        The dictionary is empty if the image is not annotated with covid mask.
        """
        return {
            path.parent.name: load(path)
            for path in self.root.glob(f'covid_masks/{id_}/*/covid_mask.npy')
        }

    def has_covid_masks(self, id_):
        return bool(list(self.root.glob(f'covid_masks/{id_}/*/covid_mask.npy')))

    def load_covid_label(self, id_):
        return load(self.root / 'covid_labels' / id_ / 'covid_label.json')

    def load_lungs_mask(self, id_):
        return load(self.root / 'lungs_masks' / id_ / 'lungs_mask.npy')

    def has_lungs_mask(self, id_):
        return (self.root / 'lungs_masks' / id_ / 'lungs_mask.npy').exists()


class RescaleToPixelSpacing(Proxy):
    def __init__(self, shadowed, pixel_spacing):
        super().__init__(shadowed)
        self.pixel_spacing = np.broadcast_to(pixel_spacing, 2).astype(float)

    def _get_scale_factor(self, id_):
        pixel_spacing = self.load_spacing(id_)[:2]
        return pixel_spacing / self.pixel_spacing

    def load_image(self, id_):
        image = self._shadowed.load_image(id_)
        scale_factor = self._get_scale_factor(id_)
        return zoom(image, scale_factor, axis=(0, 1), fill_value=np.min)

    def load_covid_masks(self, id_):
        covid_masks = self._shadowed.load_covid_masks(id_)
        scale_factor = self._get_scale_factor(id_)
        return dmap(zoom, covid_masks, scale_factor, axis=(0, 1), order=0)

    def load_lungs_mask(self, id_):
        lungs_mask = self._shadowed.load_lungs_mask(id_)
        scale_factor = self._get_scale_factor(id_)
        return zoom(lungs_mask, scale_factor, axis=(0, 1), order=0)


class CropToLungs(Proxy):
    def load_image(self, id_):
        image = self._shadowed.load_image(id_)
        lungs_mask = self._shadowed.load_lungs_mask(id_)

        if image.shape != lungs_mask.shape:
            raise RuntimeError(f'{id_}: image and lungs mask have different shapes: {image.shape} != {lungs_mask.shape}.')

        if not lungs_mask.any():
            raise RuntimeError(f'{id_}: empty lungs mask.')

        lungs_box = mask2bounding_box(lungs_mask)
        return crop_to_box(image, lungs_box)

    def load_covid_masks(self, id_):
        covid_masks = self._shadowed.load_covid_masks(id_)
        lungs_mask = self._shadowed.load_lungs_mask(id_)

        if not all(covid_mask.shape == lungs_mask.shape for covid_mask in covid_masks.values()):
            raise RuntimeError(f'{id_}: covid and lungs masks have different shapes.')

        if not lungs_mask.any():
            raise RuntimeError(f'{id_}: empty lungs mask.')

        lungs_box = mask2bounding_box(lungs_mask)
        return dmap(crop_to_box, covid_masks, lungs_box)

    def load_lungs_mask(self, id_):
        lungs_mask = self._shadowed.load_lungs_mask(id_)

        if not lungs_mask.any():
            raise RuntimeError(f'{id_}: empty lungs mask.')

        lungs_box = mask2bounding_box(lungs_mask)
        return crop_to_box(lungs_mask, lungs_box)


class NormalizeIntensities(Proxy):
    def __init__(self, shadowed, window):
        super().__init__(shadowed)
        self.window = window

    def load_image(self, id_):
        image = self._shadowed.load_image(id_)
        return normalize_intensities(image, *self.window)
