from pathlib import Path
from typing import Tuple, Optional
import numpy as np
from tqdm import trange


def write_ply(
    filename: Path,
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    normals: Optional[np.ndarray] = None,
    comment: Optional[str] = None,
    *,
    verbose: bool = False,
):
    """Write a point cloud to a .ply file.

    Parameters
    ----------
    filename : Path
        Path to the .ply file.
    points : np.ndarray
        Point cloud of shape (N, 3), dtype=np.float32.
    colors : np.ndarray, optional
        Colors of shape (N, 3), dtype=np.uint8, by default None
    normals : np.ndarray, optional
        Normals of shape (N, 3), dtype=np.float32, by default None
    comment : str, optional
        comment to be written in the header, by default None
    verbose : bool, optional
        Whether to show a progress bar, by default False
    """
    filename = Path(filename)
    if filename.suffix != ".ply":
        raise ValueError("File extension must be .ply")

    has_colors = colors is not None
    has_normals = normals is not None

    points = points.astype(np.float32)

    if has_colors:
        colors = colors.astype(np.uint8)
        if colors.shape != points.shape:
            raise ValueError("Colors must have the same shape as points")

    if has_normals:
        normals = normals.astype(np.float32)
        if normals.shape != points.shape:
            raise ValueError("Normals must have the same shape as points")
        
    filename.parent.mkdir(parents=True, exist_ok=True)
    with open(filename, "wb") as f:
        f.write(b"ply\n")
        f.write(b"format binary_little_endian 1.0\n")
        f.write(b"element vertex %d\n" % len(points))

        if comment is not None:
            for com in comment.split("\n"):
                f.write(b"comment %s\n" % com.encode("utf-8"))

        f.write(b"property float x\n")
        f.write(b"property float y\n")
        f.write(b"property float z\n")

        if has_colors:
            f.write(b"property uchar red\n")
            f.write(b"property uchar green\n")
            f.write(b"property uchar blue\n")

        if has_normals:
            f.write(b"property float nx\n")
            f.write(b"property float ny\n")
            f.write(b"property float nz\n")

        f.write(b"end_header\n")

        progress = trange(len(points), desc=f"Writing {filename}", leave=False, disable=not verbose)
        for i in progress:
            f.write(points[i])
            if has_colors:
                f.write(colors[i])
            if has_normals:
                f.write(normals[i])

        f.write(b"\n")


def read_ply(filename: Path, *, verbose: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Read a point cloud from a .ply file.

    Parameters
    ----------
    filename : Path
        Path to the .ply file.
    verbose : bool, optional
        Whether to show a progress bar, by default False

    Returns
    -------
    points : np.ndarray
        Point cloud of shape (N, 3), dtype=np.float32.
    colors : np.ndarray
        Colors of shape (N, 3), dtype=np.uint8. None if there are no colors.
    normals : np.ndarray
        Normals of shape (N, 3), dtype=np.float32. None if there are no normals.
    comment : str
        comment in the header. None if there are no comment.
    """
    filename = Path(filename)
    if filename.exists() is False:
        raise FileNotFoundError(f"{filename} does not exist")

    with open(filename, "rb") as f:
        # Read header
        header = []
        while True:
            line = f.readline().decode("utf-8")
            header.append(line)
            if line == "end_header\n":
                break

        # Element vertex
        num = int([line.split(" ")[-1] for line in header if "element vertex" in line][0])

        # comment
        has_comment = any("comment" in line for line in header)
        if has_comment:
            comment = [line.split(" ", 1)[-1] for line in header if "comment" in line]
            comment = "".join(comment)[:-1]  # [:-1] to remove "\n" at the end
        else:
            comment = None

        # Properties
        properties = [line.split(" ")[-1].replace("\n", "") for line in header if "property" in line]

        # Check if there are colors and normals
        has_colors = "red" in properties and "green" in properties and "blue" in properties
        has_normals = "nx" in properties and "ny" in properties and "nz" in properties

        # Read points and colors
        points = np.empty((num, 3), dtype=np.float32)
        colors = np.empty((num, 3), dtype=np.uint8) if has_colors else None
        normals = np.empty((num, 3), dtype=np.float32) if has_normals else None

        progress = trange(num, leave=False, desc=f"Reading {filename}", disable=not verbose)
        for i in progress:
            points[i] = np.frombuffer(f.read(12), dtype=np.float32)
            if has_colors:
                colors[i] = np.frombuffer(f.read(3), dtype=np.uint8)
            if has_normals:
                normals[i] = np.frombuffer(f.read(12), dtype=np.float32)

    return points, colors, normals, comment
