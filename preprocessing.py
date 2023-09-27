import skimage as sk
import numpy as np
from skimage import filters, exposure
from numpy.typing import ArrayLike


def gaussian_filter(image: ArrayLike, sigma: float = 1) -> ArrayLike:
    return filters.gaussian(sk.img_as_float(image), sigma)


def get_details(image: ArrayLike, sigma: float = 1) -> ArrayLike:
    details = sk.img_as_float(image) - gaussian_filter(image, sigma)
    return details


def rescale_img(T: ArrayLike) -> ArrayLike:
    T = T - T.min()
    return T / T.max()


def sharpen(image: ArrayLike, alpha: float = 1, sigma: float = 5) -> ArrayLike:
    # return rescale_img(sk.img_as_float(image) + (alpha * get_details(image, sigma)))
    return np.clip(sk.img_as_float(image) + (alpha * get_details(image, sigma)), 0, 1)


def adaptive_equalization(image: ArrayLike) -> ArrayLike:
    im_float = sk.img_as_float(image)
    return exposure.equalize_adapthist(im_float, clip_limit=0.02)


def histogram_equalization(image: ArrayLike) -> ArrayLike:
    im_float = sk.img_as_float(image)
    return exposure.equalize_hist(im_float)


def contrast_stretch(image: ArrayLike, percentile_range: tuple = (5, 95)) -> ArrayLike:
    im_float = sk.img_as_float(image)
    pmin, pmax = np.percentile(im_float, percentile_range)
    return exposure.rescale_intensity(im_float, in_range=(pmin, pmax))
