from pathlib import Path
from typing import Sequence, Union
import numpy as np
import numpy.typing as npt
import OpenEXR  # https://excamera.com/sphinx/articles-openexr.html
import Imath


def imread_exr(filename: Union[str, Path], header_names: Union[str, Sequence[str]] = ("B", "G", "R")) -> npt.NDArray[np.float32]:
    """Read OpenEXR image file

    This function is based on the following code:
    - https://gist.github.com/jadarve/de3815874d062f72eaf230a7df41771b

    Parameters
    ----------
    filename : str | Path
        Filename of OpenEXR image file
    header_names : str | Sequence[str]
        Header names of channel, by default ("B", "G", "R") (equivalent to OpenCV's imread function)

    Returns
    -------
    image : npt.NDArray[np.float32]
        Extracted image (height, width, len(header_names)). The order of the last channel follows `header_names`.

    Examples
    --------
    >>> filename = "your_image_name.exr"
    >>> img_rgb = imread_exr(filename, ("R", "G", "B"))
    >>> img_red = imread_exr(filename, "R")
    """
    filename = Path(filename)
    if not (filename.is_file() and filename.suffix == ".exr"):
        raise FileNotFoundError(f"'{filename}' must be an OpenEXR image file")

    if isinstance(header_names, str):
        header_names = [header_names]

    exrfile = OpenEXR.InputFile(str(filename))
    header = exrfile.header()

    # Size of image
    dw = header["dataWindow"]
    height = dw.max.y - dw.min.y + 1
    width = dw.max.x - dw.min.x + 1

    # Extract channels
    image = np.empty((height, width, len(header_names)), dtype=np.float32)
    all_header_names = list(header["channels"].keys())
    for i, header_name in enumerate(header_names):
        if header_name not in all_header_names:
            raise ValueError(f"'{filename}' image has no attribute '{header_name}'. Select from {all_header_names}.")

        img_1ch = exrfile.channel(header_name, Imath.PixelType(Imath.PixelType.FLOAT))
        img_1ch = np.frombuffer(img_1ch, dtype=np.float32)
        img_1ch = np.reshape(img_1ch, (height, width))

        image[..., i] = img_1ch

    image = np.squeeze(image)

    return image
