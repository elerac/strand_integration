from typing import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import numpy.typing as npt
import cv2


def imread_multiple(paths: Sequence[Path], flags: int = cv2.IMREAD_COLOR, dtype: npt.DTypeLike = None, scale: float=1.0) -> np.ndarray:
    """Read multiple images in parallel 

    Parameters
    ----------
    paths : Sequence[Path]
        Multiple image paths
    flags : int, optional
        Flag fot cv2.imread, by default cv2.IMREAD_COLOR
    dtype : npt.DTypeLike, optional
        Dtype to convert, by default None

    Returns
    -------
    images : np.ndarray
        Images (num_images, height, width, channels)

    Examples
    --------
    >>> paths = ["image1.png", "image2.png", "image3.png"]
    >>> images = imread_multiple(paths)
    >>> images.shape
    (3, 256, 256, 3)
    """
    # Determine the number of images
    num_images = len(paths)
    if num_images == 0:
        raise ValueError("No images are given.")

    def read_image(filename: Path) -> np.ndarray:
        filename = Path(filename)
        if not filename.exists():
            raise FileNotFoundError(f"{filename} does not exist.")
        img = cv2.imread(str(filename), flags)
        if scale != 1.0:
            img = cv2.resize(img, None, fx=scale, fy=scale)
        return img

    # Read the first image to determine its shape and type
    img0 = read_image(paths[0])
    if dtype is None:
        dtype = img0.dtype

    # Initialize the output array
    images = np.empty((num_images, *img0.shape), dtype=dtype)

    # Define a function to read an individual image into the output array
    def read_and_set_image(i: int) -> None:
        img = read_image(paths[i]).astype(dtype)
        images[i] = img

    # Use a thread pool to read the images in parallel
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(read_and_set_image, i) for i in range(num_images)]

        # Wait for all threads to finish
        for future in futures:
            pass

    # Return the array
    return images