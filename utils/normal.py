import warnings
from pathlib import Path
import numpy as np
import numpy.typing as npt
import cv2


def normal2bgr(img_normal: np.ndarray, dtype: npt.DTypeLike = np.uint16) -> np.ndarray:
    """Convert normal map image (float) to BGR image (uint8 or uint16)

    nx: -1 to +1 :  Red [0, max_val]
    ny: -1 to +1 :  Green [0, max_val]
    nz:  0 to -1 :  Blue [max_val / 2, max_val]

    Parameters
    ----------
    img_normal : np.ndarray
        Image of normal map (nx, ny, nz). Shape is (..., 3)
    dtype : npt.DtypeLike, optional
        `dtype` of returned image, by default np.uint16

    Returns
    -------
    img_normal_bgr : np.ndarray
        Returned image.
    """
    if dtype == np.uint8:
        max_val = 2**8 - 1
    elif dtype == np.uint16:
        max_val = 2**16 - 1
    else:
        raise TypeError(f"`dtype` must be `np.uint8` or `np.uint`6`, not {dtype}")

    img_nz = img_normal[..., 2]

    img_nx, img_ny, img_nz = cv2.split(img_normal)

    # Check value range
    if np.any(img_nx < -1.0) or np.any(1.0 < img_nx):
        warnings.warn("nx is out of range [-1, 1]")

    if np.any(img_ny < -1.0) or np.any(1.0 < img_ny):
        warnings.warn("ny is out of range [-1, 1]")

    if np.any(img_nz < -1.0) or np.any(0.0 < img_nz):
        warnings.warn("nz is out of range [-1, 0]")

    img_b = -img_nz * max_val / 2.0 + max_val / 2.0
    img_g = (img_ny + 1.0) * max_val / 2.0
    img_r = (img_nx + 1.0) * max_val / 2.0
    img_normal_bgr = cv2.merge([img_b, img_g, img_r])
    img_normal_bgr = np.clip(img_normal_bgr, 0, max_val).astype(dtype)

    return img_normal_bgr


def bgr2normal(img_normal_bgr: np.ndarray) -> np.ndarray:
    """Convert BGR image (uint8 or uint16) to normal map image (float)

    nx: -1 to +1 :  Red [0, max_val]
    ny: -1 to +1 :  Green [0, max_val]
    nz:  0 to -1 :  Blue [max_val / 2, max_val]

    Parameters
    ----------
    img_normal_bgr : np.ndarray
        Image of normal map (nx, ny, nz). Shape is (..., 3)

    Returns
    -------
    img_normal : np.ndarray
        Returned image.
    """
    dtype = img_normal_bgr.dtype
    if dtype == np.uint8:
        max_val = 2.0**8 - 1
    elif dtype == np.uint16:
        max_val = 2.0**16 - 1
    else:
        raise TypeError(f"`dtype` must be `np.uint8` or `np.uint16`, not {dtype}")

    img_b, img_g, img_r = cv2.split(img_normal_bgr)

    img_invalid = img_b < (max_val / 2.0)

    img_nx = img_r / max_val * 2.0 - 1.0
    img_ny = img_g / max_val * 2.0 - 1.0
    img_nz = -(img_b - max_val / 2.0) / max_val * 2.0

    img_normal = cv2.merge([img_nx, img_ny, img_nz])

    img_normal[img_invalid] = np.nan
    return img_normal


def imread_normal(filename: Path) -> npt.NDArray[np.float64]:
    if not filename.exists():
        raise FileNotFoundError(f"{filename} does not exist")

    img_normal_bgr = cv2.imread(str(filename), -1)[..., :3]
    img_normal = bgr2normal(img_normal_bgr)
    return img_normal


def imwrite_normal(filename: Path, img_normal: npt.NDArray[np.float64]) -> None:
    is_float = np.issubdtype(img_normal.dtype, np.floating)
    if not is_float:
        raise TypeError(f"`img_normal` must be float type, not {img_normal.dtype}")

    img_normal_bgr = normal2bgr(img_normal, dtype=np.uint16)

    img_b, img_g, img_r = cv2.split(img_normal_bgr)
    max_val = 2.0**16 - 1
    img_invalid = img_b < (max_val / 2.0)

    img_normal_bgra = cv2.cvtColor(img_normal_bgr, cv2.COLOR_BGR2BGRA)
    img_normal_bgra[img_invalid, 3] = 0  # alpha

    cv2.imwrite(str(filename), img_normal_bgra)
