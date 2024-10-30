import argparse
from pathlib import Path
import time
import numpy as np
import cv2
import open3d as o3d


def capture_screen_with_alpha(vis: o3d.visualization.Visualizer) -> np.ndarray:
    """Capture screen with alpha channel in Open3D

    Parameters
    ----------
    vis : o3d.visualization.Visualizer
        Visualizer

    Returns
    -------
    np.ndarray
        Image with alpha channel (BGRA)
    """
    render_option = vis.get_render_option()

    background_color_default = render_option.background_color

    # Black background
    render_option.background_color = (0, 0, 0)
    img_screenshot_black = np.array(vis.capture_screen_float_buffer(do_render=True))

    # White background
    render_option.background_color = (1, 1, 1)
    img_screenshot_white = np.array(vis.capture_screen_float_buffer(do_render=True))

    # Restore background color
    render_option.background_color = background_color_default

    # Alpha
    img_diff = img_screenshot_white - img_screenshot_black
    img_alpha = 255 - np.clip(np.mean(img_diff, axis=-1) * 255, 0, 255).astype(np.uint8)

    height, width = img_screenshot_white.shape[:2]
    img_screenshot_alpha = np.empty((height, width, 4), dtype=np.uint8)
    img_screenshot_alpha[..., :3] = np.clip(img_screenshot_white[..., ::-1] * 255, 0, 255).astype(np.uint8)
    img_screenshot_alpha[..., 3] = img_alpha

    return img_screenshot_alpha


def main():
    parser = argparse.ArgumentParser("Capture rendered images from ply files")
    parser.add_argument("input", type=Path, nargs="+", help="Input ply files (determine camera position interactively with first ply file and apply it to all)")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output directory (default: same as input directory)")
    parser.add_argument("--size", type=int, nargs=2, default=(600, 900), help="Window size (width, height)")
    parser.add_argument("--gamma", type=str, default="1.0", help="Gamma correction for color (e.g., 1/2.2, 1/1.8)")
    parser.add_argument("--near", type=float, default=0.00001, help="Near plane")
    parser.add_argument("--far", type=float, default=100000, help="Far plane")
    args = parser.parse_args()

    # Get time
    timestr = time.strftime("%Y-%m-%d-%H-%M-%S")
    print("Time: ", timestr)

    # Check input
    input = args.input
    num = len(input)
    print("Input ply files:")
    for i, filename in enumerate(input):
        if not (filename.exists() and filename.suffix == ".ply"):
            raise FileNotFoundError(f"Ply file not found: '{filename}'")

        print(f"  [{i+1}/{num}] '{filename}'")

    # Gamma value
    gamma = eval(args.gamma)
    print(f"Gamma value: {gamma:.3f}")

    # Window size
    width, height = args.size
    print(f"Window size: {width} x {height}")

    # Init visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, left=0, top=0)
    render_option = vis.get_render_option()
    view_control = vis.get_view_control()
    render_option.point_size = 1.0
    view_control.set_constant_z_far(args.far)
    view_control.set_constant_z_near(args.near)

    for i, filename_ply in enumerate(input):
        print(f"----- [{i+1}/{num}] -----")
        # Read ply file
        print(f"PLY file: '{filename_ply}'")
        pcd = o3d.io.read_point_cloud(str(filename_ply))
        print(f"Number of points: {len(pcd.points)}")

        # Gamma correction
        pcd.colors = o3d.utility.Vector3dVector(np.array(pcd.colors) ** gamma)

        # Remove normals
        if len(pcd.normals) > 0:
            pcd.normals = o3d.utility.Vector3dVector(np.zeros((0, 3)))

        # Add geometry
        reset_bounding_box = True if i == 0 else False
        vis.add_geometry(pcd, reset_bounding_box=reset_bounding_box)

        if i == 0:
            print()
            print("Move camera to desired position and press [q] to capture screenshot for all ply files")
            vis.run()
            camera_params = vis.get_view_control().convert_to_pinhole_camera_parameters()

        # Capture screenshot
        img_screenshot = capture_screen_with_alpha(vis)

        # Save screenshot
        if args.output is None:
            output = filename_ply.parent
        else:
            output = args.output / filename_ply.parent
        output.mkdir(parents=True, exist_ok=True)

        filename_png = filename_ply.with_suffix(".png")
        filename_png = output / f"Screenshot_{timestr}_{filename_png.name}"

        print(f"Save screenshot: '{filename_png}'")
        cv2.imwrite(str(filename_png), img_screenshot)

        # Remove geometry
        vis.clear_geometries()

    vis.destroy_window()

    print()
    print("Camera parameters:")
    print(camera_params.intrinsic.intrinsic_matrix)
    print(camera_params.extrinsic)


if __name__ == "__main__":
    main()