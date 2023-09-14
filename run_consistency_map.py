import argparse
from pathlib import Path
import time
import warnings
from tqdm import trange, tqdm
import numpy as np
import cv2
import strandtools
import utils
from utils import configs


def main():
    parser = argparse.ArgumentParser("Gnerate 3D line consistency map")
    parser.add_argument("input", type=Path, help="Path to the input directory of multi-view data (e.g., result/lpmvs/straight)")
    parser.add_argument("-o", "--output", type=Path, help="Output directory (default: result/consistency/<input>)")
    parser.add_argument("--views", type=int, nargs="+", default=None, help="Specify the view indices to be processed (default: all views)")
    parser.add_argument("--num_neighbors", type=int, default=10)
    parser.add_argument("--num_least_consisten_neigbor", type=int, default=3)
    parser.add_argument("--sigma", type=float, default=10.0, help="Sigma of Gaussian kernel")
    parser.add_argument("--write_ply", action="store_true", help="Write ply file")
    parser.add_argument("--imshow", action="store_true")
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=RuntimeWarning)

    multiviewdata = utils.read_multiview(args.input, read_images=False, verbose=False)

    # Create output directory
    if args.output is None:
        output = Path("result") / Path("consistency") / args.input.name
    else:
        output = args.output
    output.mkdir(parents=True, exist_ok=True)

    info = ""
    info += f"Datetime: {time.strftime('%Y/%m/%d %H:%M:%S')}\n"
    info += f"Input: {args.input}\n"
    info += f"Output: {output}\n"
    info += f"Number of views: {len(multiviewdata)}\n"
    info += f"Number of neighbors: {args.num_neighbors}\n"
    info += f"Number of least consistent neighbors: {args.num_least_consisten_neigbor}\n"
    info += f"Sigma: {args.sigma}\n"

    print("Start generating depth consistency map")
    print(info)

    # Save parameters
    with open(output / "params.txt", "w") as f:
        f.write(info)

    if args.views is None:
        progress_bar = trange(len(multiviewdata))
    else:
        progress_bar = tqdm(args.views)

    for view_i in progress_bar:
        progress_bar.set_description(f"View {view_i}")

        reference_view = utils.read_multiview(args.input, view_select=[view_i], verbose=False)[0]
        neighbor_indices = multiviewdata.get_neighbor_index_vector(view_i, args.num_neighbors)

        width, height = reference_view.size()
        imgs_position_diff = np.empty((args.num_neighbors, height, width), dtype=np.float32)
        imgs_angle_diff = np.empty((args.num_neighbors, height, width), dtype=np.float32)
        for i, neighbor_view_i in enumerate(neighbor_indices):
            # Read a neighbor view
            neighbor_views = utils.read_multiview(args.input, view_select=[neighbor_view_i], verbose=False)[0]

            # Compute difference between reference view and neighbor view in terms of position and angle
            img_position_diff, img_angle_diff = strandtools.eval_consisntency(reference_view, neighbor_views)

            imgs_position_diff[i] = img_position_diff
            imgs_angle_diff[i] = img_angle_diff

        # Count sum of nan pixels
        imgs_isnan = np.isnan(imgs_position_diff)
        img_sum_nan = np.sum(imgs_isnan, axis=0)
        img_valid_sum = args.num_neighbors - img_sum_nan

        # Average position difference with weight (weight is the angle difference)
        imgs_weight = np.abs(np.pi / 2 - imgs_angle_diff)
        img_position_diff_average = np.nansum(imgs_position_diff**2 * imgs_weight, axis=0) / np.nansum(imgs_weight, axis=0)

        # Mask out pixels that have small number of valid neighbors because they are not reliable
        img_position_diff_average[img_valid_sum < args.num_least_consisten_neigbor] = np.nan

        # Fill NaN pixels with the maximum value
        img_mask_nan = np.isnan(img_position_diff_average)

        # We define the high depth consistency as the low position difference.
        # We use gaussian function as RBF kernel.
        img_depth_consistency = np.exp(-(img_position_diff_average) / (2 * args.sigma**2))

        img_depth_consistency[img_mask_nan] = 0.0

        # Show image
        if args.imshow:
            cv2.imshow("depth_consistency", img_depth_consistency)
            cv2.imshow("img_mask_nan", img_mask_nan.astype(np.uint8) * 255)
            key = cv2.waitKey(30)
            if key == ord("q"):
                break

        # Get point cloud amd use the depth consistency as the color (for visualization)
        img_mask_new = np.bitwise_and(np.logical_not(img_mask_nan), reference_view.img_mask > 0)
        reference_view.img_mask = img_mask_new.astype(np.uint8) * 255
        points, _ = reference_view.getDirectionalPoint()
        colors = np.clip(img_depth_consistency[img_mask_new] * 255, 0, 255).astype(np.uint8)
        colors = np.stack([colors, colors, colors], axis=-1)

        # Export
        output_sub = output / f"{view_i:02}"
        output_sub.mkdir(parents=True, exist_ok=True)
        progress_bar.set_description(f"Export to {output_sub}")
        if args.write_ply:
            utils.write_ply(output_sub / configs.filename_pointcloud, points, colors)
        cv2.imwrite(str(output_sub / configs.filename_consistency), img_depth_consistency.astype(np.float32))


if __name__ == "__main__":
    main()
