from typing import Sequence, Optional, List
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm
import strandtools
from . import configs
from .ply import write_ply
from .colormap import apply_color_to_orientation2d, apply_color_to_confidence, apply_color_to_depth, apply_color_to_direction
from .parallel_imread import imread_multiple


def read_singleview(path: Path, *, read_images: bool = True) -> strandtools.SingleViewData:
    """Read single view data from a directory.

    Parameters
    ----------
    path : Path
        Path of the directory
    read_images : bool, optional
        Whether to read images, by default True. If False, only camera parameters are read.

    Returns
    -------
    view : strandtools.SingleViewData
        Single view data
    """
    # Check if the directory exists
    path = Path(path)
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"{path} directory does not exist.")

    # Create a view
    view = strandtools.SingleViewData()

    # Camera parameters
    K = np.loadtxt(path / configs.filename_K, dtype=np.float32)  # 3x3
    R = np.loadtxt(path / configs.filename_R, dtype=np.float32)  # 3x3
    t = np.loadtxt(path / configs.filename_t, dtype=np.float32)  # 3x1
    camera = strandtools.Camera(K, R, t)
    view.camera = camera

    # max_depth and min_depth
    filename_min_depth = path / configs.filename_min_depth
    if filename_min_depth.exists():
        view.min_depth = np.loadtxt(filename_min_depth, dtype=np.float32)
    filename_max_depth = path / configs.filename_max_depth
    if filename_max_depth.exists():
        view.max_depth = np.loadtxt(filename_max_depth, dtype=np.float32)

    if not read_images:
        return view

    # Intensity image
    filename_intensity = path / configs.filename_intensity
    if filename_intensity.exists():
        img_intensity = cv2.imread(str(filename_intensity), -1)
        if img_intensity.ndim == 3:
            img_intensity = cv2.cvtColor(img_intensity, cv2.COLOR_BGR2GRAY)
        view.img_intensity = img_intensity
    else:
        # Photometric images as intensity
        filename_photometric_list = sorted(path.glob("*-[0-9][0-9]-[0-9][0-9].exr"))
        if len(filename_photometric_list) > 0:
            imlist = imread_multiple(filename_photometric_list, cv2.IMREAD_UNCHANGED, dtype=np.float32)
            img_photometric = np.mean(imlist, axis=0)
            img_photometric = cv2.cvtColor(img_photometric, cv2.COLOR_BGR2GRAY)
            view.img_intensity = img_photometric

    # 2D orientation and confidence map
    filename_orientation = path / configs.filename_orientation2d
    filename_confidence = path / configs.filename_confidence
    if filename_orientation.exists() and filename_confidence.exists():
        # If both files exist, read them
        view.img_orientation2d = cv2.imread(str(filename_orientation), -1)
        view.img_confidence = cv2.imread(str(filename_confidence), -1)
    # else:
    #     # If not, generate them
    #     img_orientation2d, img_confidence = strandtools.generate_orientation_map(view.img_intensity, num=180)
    #     view.img_orientation2d = img_orientation2d
    #     view.img_confidence = img_confidence

    # Depthmap and direction if available
    filename_depth = path / configs.filename_depth
    filename_direction = path / configs.filename_direction
    if filename_depth.exists() and filename_direction.exists():
        img_depth = cv2.imread(str(filename_depth), -1)
        if img_depth.ndim == 3:
            img_depth = img_depth[:, :, 0]
        img_direction = cv2.imread(str(filename_direction), -1)
        view.set_line(img_depth, img_direction)
    elif filename_depth.exists() and not filename_direction.exists():
        img_depth = cv2.imread(str(filename_depth), -1)
        if img_depth.ndim == 3:
            img_depth = img_depth[:, :, 0]
        img_direction = np.random.rand(*img_depth.shape, 3).astype(np.float32)
        img_direction /= np.linalg.norm(img_direction, axis=2, keepdims=True)
        view.set_line(img_depth, img_direction)
    elif not filename_depth.exists() and filename_direction.exists():
        img_direction = cv2.imread(str(filename_direction), -1)
        img_depth = np.full(img_direction.shape[:2], 1, dtype=np.float32)
        view.set_line(img_depth, img_direction)
    else:
        # Set line randomly
        width, height = view.size()
        img_depth = np.full(view.img_intensity.shape, 1, dtype=np.float32)
        img_direction = np.random.rand(*img_depth.shape, 3).astype(np.float32)
        img_direction /= np.linalg.norm(img_direction, axis=2, keepdims=True)
        view.set_line(img_depth, img_direction)

    # Mask
    filename = path / configs.filename_mask
    if filename.exists():
        img_mask = cv2.imread(str(filename), 0)
        view.img_mask = img_mask
    else:
        width, height = view.size()
        view.img_mask = np.full((height, width), 255, dtype=np.uint8)

    return view


def get_multiview_dirs(path: Path, view_select: Optional[Sequence[int]] = None) -> List[Path]:
    """Get names of subdirectories in a directory containing multiview data.

    Parameters
    ----------
    path : Path
        Path to the directory. The directory must contain subdirectories named 00, 01, 02, ...
    view_select : Optional[Sequence[int]], optional
        Indices of subdirectories to read, by default None. If None, all subdirectories are selected.

    Returns
    -------
    List[Path]
        List of subdirectories
    """
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"{path} is not a directory.")

    if view_select is None:
        subdir_list = sorted(path.glob("[0-9][0-9]"))
    else:
        subdir_list = [path / f"{i:02d}" for i in view_select]

    if len(subdir_list) == 0:
        raise FileNotFoundError(f"No subdirectories found in {path}.")

    return subdir_list


def read_multiview(data_dir: Path, view_select: Optional[Sequence[int]] = None, *, verbose: bool = False, **kwargs) -> strandtools.MultiViewData:
    """Read multiview data from a directory.

    Parameters
    ----------
    data_dir : Path
        Path to the directory. The directory must contain subdirectories named 00, 01, 02, ...
    view_select : Optional[Sequence[int]], optional
        List of view IDs to be selected from the directory. By default, all views are selected.
    verbose : bool, optional
        If True, print messages, by default False

    Returns
    -------
    strandtools.MultiViewData
        Multiview data
    """
    # Check the directory exists
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"{data_dir} does not exist.")

    # Get subdirectories
    # /path/to/data/00
    # /path/to/data/01
    # /path/to/data/02
    subdir_list = get_multiview_dirs(data_dir, view_select)

    if verbose:
        # Print subdirectories
        subdir_list_name = [str(subdir) for subdir in subdir_list]
        print(f"Found {len(subdir_list)} subdirectories:")
        print(subdir_list_name)

    # Read data
    multiviewdata = strandtools.MultiViewData()
    progress_bar = tqdm(subdir_list, disable=not verbose, leave=False)
    for subdir in progress_bar:
        if verbose:
            progress_bar.set_description(f"{subdir}")

        view = read_singleview(subdir, **kwargs)
        multiviewdata.append(view)

    return multiviewdata


def write_singleview(path: Path, view: strandtools.SingleViewData, with_extra: bool = True) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # Write intensity image
    filename_intensity = path / configs.filename_intensity
    img_intensity = view.img_intensity
    if img_intensity.shape != (0, 0):
        cv2.imwrite(str(filename_intensity), view.img_intensity)

    # Write orientation and confidence map
    filename_orientation = path / configs.filename_orientation2d
    img_orientation2d = view.img_orientation2d
    if img_orientation2d.shape != (0, 0):
        cv2.imwrite(str(filename_orientation), view.img_orientation2d)

    filename_confidence = path / configs.filename_confidence
    img_confidence = view.img_confidence
    if img_confidence.shape != (0, 0):
        cv2.imwrite(str(filename_confidence), view.img_confidence)

    # Write depthmap and direction
    filename_depthmap = path / configs.filename_depth
    img_depth = view.img_depth
    if img_depth.shape != (0, 0):
        cv2.imwrite(str(filename_depthmap), view.img_depth)

    filename_direction = path / configs.filename_direction
    img_direction = view.img_direction
    if img_direction.shape != (0, 0):
        cv2.imwrite(str(filename_direction), view.img_direction)

    # Write mask
    filename_mask = path / configs.filename_mask
    img_mask = view.img_mask
    if img_mask.shape != (0, 0):
        cv2.imwrite(str(filename_mask), view.img_mask)

    # Write K, R, t
    filename_K = path / configs.filename_K
    filename_R = path / configs.filename_R
    filename_t = path / configs.filename_t
    camera = view.camera
    np.savetxt(filename_K, camera.K)
    np.savetxt(filename_R, camera.R)
    np.savetxt(filename_t, camera.t)

    # Write min_depth and max_depth
    filename_min_depth = path / configs.filename_min_depth
    filename_max_depth = path / configs.filename_max_depth
    np.savetxt(filename_min_depth, [view.min_depth])
    np.savetxt(filename_max_depth, [view.max_depth])

    if with_extra:
        # Save the point cloud
        points, directions = view.getDirectionalPoint()
        filename_pointcloud = path / Path(configs.filename_pointcloud)
        colors = np.clip(np.abs(directions) * 255.0, 0, 255).astype(np.uint8)  # x: red, y: green, z: blue
        write_ply(filename_pointcloud, points, colors, directions)

        # Visualized orientation map
        if img_orientation2d.shape != (0, 0):
            img_orientation2d_colored = apply_color_to_orientation2d(img_orientation2d)
            filename_orientation2d_colored = configs.filename_orientation2d_colored
            cv2.imwrite(str(path / filename_orientation2d_colored), img_orientation2d_colored)

        # Visualized confidence maps
        if img_confidence.shape != (0, 0):
            img_confidence_colored = apply_color_to_confidence(img_confidence)
            filename_confidence_colored = configs.filename_confidence_colored
            cv2.imwrite(str(path / filename_confidence_colored), img_confidence_colored)

        # Visualized depthmap
        if img_depth.shape != (0, 0):
            img_depth_colored = apply_color_to_depth(img_depth, view.min_depth, view.max_depth)
            filename_depth_colored = configs.filename_depth_colored
            cv2.imwrite(str(path / filename_depth_colored), img_depth_colored)

        # Visualized direction
        if img_direction.shape != (0, 0):
            img_direction_colored = apply_color_to_direction(img_direction)
            filename_direction = configs.filename_direction_colored
            cv2.imwrite(str(path / filename_direction), img_direction_colored)


def imshow_singleview(view: strandtools.SingleViewData, delay: int = 10) -> int:
    if view.size() == (0, 0):
        return -1

    img_intensity = view.img_intensity
    img_mask = view.img_mask
    img_confidence = view.img_confidence

    # Apply color map
    img_orientation2d_colored = apply_color_to_orientation2d(view.img_orientation2d)
    img_confidence_colored = apply_color_to_confidence(img_confidence)

    cv2.imshow("img_intensity", img_intensity)
    cv2.imshow("img_mask", img_mask)
    cv2.imshow("img_orientation", img_orientation2d_colored)
    cv2.imshow("img_confidence", img_confidence_colored)
    key = cv2.waitKey(delay)
    return key


def imshow_multiview(multiview: strandtools.MultiViewData, delay: int = 100) -> int:
    key = -1
    for view in multiview:
        key_tmp = imshow_singleview(view, delay)
        if key_tmp != -1:
            key = key_tmp
    
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    return key