# Default configurations
from pathlib import Path

# Filename of the SingleViewData
filename_intensity = "intensity.exr"
filename_orientation2d = "orientation2d.exr"
filename_confidence = "confidence.exr"
filename_mask = "mask.png"
filename_depth = "depth.exr"
filename_direction = "direction.exr"
filename_K = "K.txt"
filename_R = "R.txt"
filename_t = "t.txt"
filename_min_depth = "min_depth.txt"
filename_max_depth = "max_depth.txt"

# Filename of the colored SingleViewData
filename_orientation2d_colored = f"{Path(filename_orientation2d).stem}_colored.png"
filename_confidence_colored = f"{Path(filename_confidence).stem}_colored.png"
filename_direction_colored = f"{Path(filename_direction).stem}_colored.png"
filename_depth_colored = f"{Path(filename_depth).stem}_colored.png"

# Some extra filenames
filename_pointcloud = "pointcloud.ply"
filename_consistency = "consistency.exr"
filename_absdiff = "absdiff.exr"
filename_absdiff_colored = "absdiff_colored.png"
filename_depth_colored = "depth_colored.png"
filename_error_json = "error.json"
filename_normal = "normal.png"
