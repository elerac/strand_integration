import argparse
from pathlib import Path
import math
import time
import numpy as np
import cv2
import strandtools
import utils
from tqdm import trange


def main():
    parser = argparse.ArgumentParser("3D Line Filtering")
    parser.add_argument("input", type=Path, help="Path to the input directory of multi-view data")
    parser.add_argument("output", type=Path, help="Filename of the output ply file")
    parser.add_argument("--num_neighbors", type=int, default=6)
    parser.add_argument("--num_least_consisten_neigbor", type=int, default=2)
    parser.add_argument("-tp", "--thresh_position", type=float, default=2.7)
    parser.add_argument("-td", "--thresh_angle", type=float, default=10.0, help="degree")
    parser.add_argument("--ratio", type=float, default=1.0)
    parser.add_argument("--imshow", action="store_true")
    args = parser.parse_args()

    np.random.seed(0)

    if args.output.suffix != ".ply":
        raise ValueError(f"Output must be a ply file, but got {args.output}")

    multiviewdata = utils.read_multiview(args.input, read_images=False, verbose=False)

    info = ""
    info += f"Datetime: {time.strftime('%Y/%m/%d %H:%M:%S')}\n"
    info += f"Input: {args.input}\n"
    info += f"Output: {args.output}\n"
    info += f"Number of views: {len(multiviewdata)}\n"
    info += f"Number of neighbors: {args.num_neighbors}\n"
    info += f"Number of least consistent neighbors: {args.num_least_consisten_neigbor}\n"
    info += f"Threshold of position difference: {args.thresh_position}\n"
    info += f"Threshold of angle difference (degree): {args.thresh_angle}\n"
    info += f"Ratio of points: {args.ratio}\n"

    print("Start 3D line filtering")
    print(info)

    apply_filtering = args.thresh_position != np.inf or args.thresh_angle != np.inf

    num_points_origin_list = []
    num_points_filtered_list = []

    points_list = []
    directions_list = []
    progress_bar = trange(len(multiviewdata))
    for view_i in progress_bar:
        progress_bar.set_description(f"View {view_i}")

        reference_view = utils.read_multiview(args.input, view_select=[view_i])[0]
        neighbor_indices = multiviewdata.get_neighbor_index_vector(view_i, args.num_neighbors)

        if apply_filtering:
            width, height = reference_view.size()
            imgs_position_diff = np.empty((args.num_neighbors, height, width), dtype=np.float32)
            imgs_angle_diff = np.empty((args.num_neighbors, height, width), dtype=np.float32)
            for i, neighbor_view_i in enumerate(neighbor_indices):
                # Read a neighbor view
                neighbor_views = utils.read_multiview(args.input, view_select=[neighbor_view_i])[0]

                # Compute difference between reference view and neighbor view in terms of position and angle
                img_position_diff, img_angle_diff = strandtools.eval_consisntency(reference_view, neighbor_views)

                imgs_position_diff[i] = img_position_diff
                imgs_angle_diff[i] = img_angle_diff

            # Check consistency
            imgs_position_valid = imgs_position_diff < args.thresh_position
            imgs_angle_valid = imgs_angle_diff < math.radians(args.thresh_angle)
            imgs_line_valid = np.bitwise_and(imgs_position_valid, imgs_angle_valid)
            img_line_valid = np.sum(imgs_line_valid, axis=0) >= args.num_least_consisten_neigbor

            # Update mask
            img_mask = reference_view.img_mask > 0
            img_mask_new = np.bitwise_and(img_mask, img_line_valid)
            reference_view.img_mask = img_mask_new.astype(np.uint8) * 255

            # Count number of points
            num_points_origin = np.sum(img_mask)
            num_points_filtered = np.sum(img_mask_new)
            num_points_origin_list.append(num_points_origin)
            num_points_filtered_list.append(num_points_filtered)

        # Get points and directions
        points, directions = reference_view.getDirectionalPoint()

        # Randomly select points
        if 0 < args.ratio < 1.0:
            num_points = len(points)
            num_sample = int(num_points * args.ratio)
            indices = np.random.choice(num_points, num_sample, replace=False)
            points = points[indices]
            directions = directions[indices]

        points_list.append(points)
        directions_list.append(directions)

        # Show mask image
        if args.imshow:
            img_mask = reference_view.img_mask > 0
            cv2.imshow(f"img_mask", img_mask.astype(np.uint8) * 255)
            key = cv2.waitKey(1)
            if key == ord("q"):
                break

    points = np.concatenate(points_list, axis=0)
    directions = np.concatenate(directions_list, axis=0)
    colors = (np.abs(directions) * 255).astype(np.uint8)
    total_num_points_origin = np.sum(num_points_origin_list)
    total_num_points_filtered = np.sum(num_points_filtered_list)
    print(f"Number of points (origin): {total_num_points_origin}")
    print(f"Number of points (filtered): {total_num_points_filtered}")
    print(f"Ratio of remained points: {total_num_points_filtered / total_num_points_origin * 100:0.2f} [%]")
    print(f"Ratio of removed points: {100.0 - total_num_points_filtered / total_num_points_origin * 100:0.2f} [%]")

    print(f"Number of points: {len(points)}")
    print("Export ply file")
    utils.write_ply(args.output, points, colors, directions, comment=info)


if __name__ == "__main__":
    main()
