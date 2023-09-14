import numpy as np
import open3d as o3d


def construct_camera(width: int, height: int, K: np.ndarray, R: np.ndarray, t: np.ndarray) -> o3d.camera.PinholeCameraParameters:
    """Camera parameter of Open3D PinholeCameraParameters"""
    camera_params = o3d.camera.PinholeCameraParameters()

    camera_params.intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, K)

    extrinsic = np.eye(4)
    extrinsic[:3, :3] = R
    extrinsic[:3, 3] = t
    camera_params.extrinsic = extrinsic

    return camera_params


def pcd_from_depthmap(depthmap: np.ndarray, camera_params: o3d.camera.PinholeCameraParameters, mask: np.ndarray) -> o3d.geometry.PointCloud:
    if depthmap.shape != mask.shape:
        raise ValueError("The shape of depthmap and mask must be same")

    depthmap[np.bitwise_not(mask)] = 1.0  # avoid invalid depth

    depthmap[depthmap < 0] = 0.001

    # Construct pcd all points
    imgo3d_depth = o3d.geometry.Image(depthmap)
    pcd = o3d.geometry.PointCloud.create_from_depth_image(imgo3d_depth, camera_params.intrinsic, camera_params.extrinsic, 1, 1)

    # Extract masked points
    mask_1d = mask.flatten()
    pcd.points = o3d.utility.Vector3dVector(np.array(pcd.points)[mask_1d])

    return pcd


def capture_screen_with_alpha(vis: o3d.visualization.Visualizer) -> np.ndarray:
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


def merge_alpha(img_rgb: np.ndarray, img_alpha: np.ndarray) -> np.ndarray:
    if img_rgb.shape[:2] != img_alpha.shape[:2]:
        raise ValueError("The height and width of two images must be same")

    img_rgba = np.empty((*img_rgb.shape[:2], 4), dtype=img_rgb.dtype)
    img_rgba[..., :3] = img_rgb
    img_rgba[..., 3] = img_alpha.astype(img_rgba.dtype)

    return img_rgba