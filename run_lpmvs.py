import argparse
from pathlib import Path
import time
import numpy as np
from tqdm import tqdm, trange
import strandtools
from utils import read_multiview, write_singleview, imshow_multiview


def main():
    parser = argparse.ArgumentParser("Run LPMVS")
    # inputx
    parser.add_argument("input", type=Path)
    # LPMVS
    parser.add_argument("--num_neighbors", type=int, default=5)
    parser.add_argument("--radius", type=float, default=10.0)
    parser.add_argument("--num_samples", type=int, default=41)
    parser.add_argument("--depth_perturbation", type=float, default=1.0)
    parser.add_argument("--direction_perturbation", type=float, default=0.01)
    parser.add_argument("--max_iter", type=int, default=25)
    parser.add_argument("--min_depth", type=float, default=100.0)
    parser.add_argument("--max_depth", type=float, default=255.0)
    # output
    parser.add_argument("-o", "--output", type=Path)
    # others
    parser.add_argument("--views", type=int, nargs="+", default=None, help="Specify the view indices to be processed (default: all views)")
    parser.add_argument("--imshow", action="store_true")
    parser.add_argument("--on_memory", action="store_true", help="Read all images to memory")
    parser.add_argument("--scale", type=float, default=1.0, help="Resize image for faster processing. Debug purpose.")
    parser.add_argument("--save_intermediate", action="store_true", help="Save intermediate results")
    args = parser.parse_args()

    print(f"Read multi-view data from {args.input}")
    if args.on_memory:
        # Read all images (including unused images)
        # Require large memory.
        multiviewdata = read_multiview(args.input, verbose=True)
        if args.scale != 1.0:
            for view in multiviewdata:
                view.rescale(args.scale)
    else:
        # Read minimal images (defined by --num_neighbors) for each view
        # Memory efficient, but slow because of disk I/O for every iteration.
        # In the following line, only camera parameters are read.
        multiviewdata = read_multiview(args.input, read_images=False, verbose=True)
       
    if args.imshow:
        print("Show multi-view data...")
        imshow_multiview(multiviewdata)

    print(f"Multi-view data is ready. Number of views: {len(multiviewdata)}")
    print(f"shape of image: {multiviewdata[0].img_intensity.shape}")

    # Parameters
    num_neighbors = args.num_neighbors
    radius = args.radius
    num_samples = args.num_samples
    depth_perturbation = args.depth_perturbation
    direction_perturbation = args.direction_perturbation
    max_iter = args.max_iter
    min_depth = args.min_depth
    max_depth = args.max_depth

    # Output directory
    if args.output is None:
        output = Path("result") / Path("lpmvs") / args.input.name
    else:
        output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    # Save the input parameters
    info = f"Date: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n"
    info += f"Input directory: {args.input}\n"
    info += f"Output directory: {output}\n"
    info += f"Number of neighbors: {num_neighbors}\n"
    info += f"Radius: {radius}\n"
    info += f"Number of samples: {num_samples}\n"
    info += f"Depth perturbation: {depth_perturbation}\n"
    info += f"Direction perturbation: {direction_perturbation}\n"
    info += f"Max iteration: {max_iter}\n"
    info += f"Min depth: {min_depth}\n"
    info += f"Max depth: {max_depth}\n"

    with open(output / "params.txt", "w") as f:
        f.write(info)

    print("Start LPMVS")
    print("--------------------------------")
    print(info, end="")
    print("--------------------------------")

    if args.views is None:  # All views
        progress_bar = trange(len(multiviewdata))
    else:  # Specified views
        progress_bar = tqdm(args.views)

    for view_i in progress_bar:
        progress_bar.set_description(f"View {view_i}")

        # Output directory
        zero_num = len(str(len(multiviewdata)))
        output_view = output / str(view_i).zfill(zero_num)

        # Get the reference view and its neighbors
        if args.on_memory:
            reference_view = multiviewdata[view_i]
            neighbor_views = multiviewdata.get_neighbor(view_i, num_neighbors)
        else:
            reference_view = read_multiview(args.input, view_select=[view_i])[0]
            neighbor_indices = multiviewdata.get_neighbor_index_vector(pos=view_i, num=args.num_neighbors)
            neighbor_views = read_multiview(args.input, view_select=neighbor_indices)
            if args.scale != 1.0:
                reference_view.rescale(args.scale)
                for view in neighbor_views:
                    view.rescale(args.scale)

        reference_view.min_depth = min_depth
        reference_view.max_depth = max_depth
        reference_view.set_random_line()

        progress_bar_sub = trange(max_iter, position=1, leave=False)
        for iter in progress_bar_sub:
            # Update the 3D line map via spatial propagation
            progress_bar_sub.set_description(f"Propagate {iter}")
            strandtools.propagate(
                reference_view,
                neighbor_views,
                radius,
                num_samples,
            )

            # Refine the 3D line map via random perturbation
            progress_bar_sub.set_description(f"Refinement {iter}")
            strandtools.refinement(
                reference_view,
                neighbor_views,
                radius,
                num_samples,
                depth_perturbation,
                direction_perturbation,
            )

            # Clip the depth map
            img_depth = np.clip(reference_view.img_depth, reference_view.min_depth, reference_view.max_depth)
            reference_view.img_depth = img_depth

            # Export results
            if args.save_intermediate or iter == max_iter - 1:
                progress_bar_sub.set_description(f"Export results {iter}")
                write_singleview(output_view, reference_view, with_extra=True)

        # Release img_line to save memory
        reference_view.release_line()

    print("Finish LPMVS")


if __name__ == "__main__":
    main()
