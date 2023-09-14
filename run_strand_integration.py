import argparse
from pathlib import Path
import time
from tqdm import tqdm, trange
import numpy as np
import cv2
import torch
import utils
from utils import configs
from utils.torch_utils import auto_torch_device, WeightedMean


def main():
    parser = argparse.ArgumentParser("Strand integration")
    parser.add_argument("input", type=Path, help="Input directory")
    parser.add_argument("--consistency", type=Path, default=None, help="Input directory for consistency term")
    parser.add_argument("--normal", type=Path, default=None, help="Input directory for normal term")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output directory")
    parser.add_argument("--max_iter", type=int, default=30000, help="Number of iterations")
    parser.add_argument("--lr", type=float, default=1.0, help="Initial learning rate")
    parser.add_argument("--lr_decay", type=float, default=0.01, help="The decay ratio of learning rate (last_lr = initial_lr * lr_decay)")
    parser.add_argument("-wd", "--w_direction", type=float, default=10.0, help="Weight for direction")
    parser.add_argument("-wn", "--w_normal", type=float, default=0.0, help="Weight for normal")
    parser.add_argument("--views", type=int, nargs="+", default=None, help="Specify the view indices to be processed (default: all views)")
    parser.add_argument("--device", type=str, default=auto_torch_device(), help="PyTorch device (by default, automatically select)")
    parser.add_argument("--imshow", action="store_true", help="Show images")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    args = parser.parse_args()

    device = args.device
    if args.debug:
        torch.autograd.set_detect_anomaly(True)

    # Output directory
    if args.output is None:
        output = Path("result") / Path("strand_integration") / args.input.name
    else:
        output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    # Save the input parameters
    info = f"Date: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n"
    info += f"Input: {args.input}\n"
    info += f"Output: {output}\n"
    info += f"Max iteration: {args.max_iter}\n"
    info += f"Learning rate: {args.lr}\n"
    info += f"Learning rate decay: {args.lr_decay}\n"
    info += f"Weight for normal: {args.w_normal}\n"
    info += f"Weight for direction: {args.w_direction}\n"
    info += f"Consistency term: {args.consistency}\n"
    info += f"Device: {device}\n"

    with open(output / "params.txt", "w") as f:
        f.write(info)

    print("Start strand integration")
    print("--------------------------------")
    print(info, end="")
    print("--------------------------------")

    print(f"Reading single view data from {args.input}...")

    mvs_subdirs = utils.get_multiview_dirs(args.input)
    num = len(mvs_subdirs)

    if args.consistency is not None:
        mvs_consistency_subdirs = utils.get_multiview_dirs(args.consistency)
        if len(mvs_consistency_subdirs) != num:
            raise ValueError(f"Number of views in {args.input} and {args.consistency} are different")

    if args.views is None:  # All views
        progress_bar = tqdm(range(num))
    else:  # Specified views
        progress_bar = tqdm(args.views)

    for view_i in progress_bar:
        progress_bar.set_description(f"View {view_i}")

        input = mvs_subdirs[view_i]
        view = utils.read_singleview(input)

        # Camera parameters
        K = view.camera.K
        R = view.camera.R
        fx, fy = K[0, 0], K[1, 1]
        width, height = view.size()

        # Depth
        depth_prior = view.img_depth

        # Mask (boarder is False)
        mask_prior = view.img_mask > 0
        mask_prior[0, :] = False
        mask_prior[-1, :] = False
        mask_prior[:, 0] = False
        mask_prior[:, -1] = False

        # Normal
        if args.w_normal > 0:
            if args.normal is not None:
                input_normal = utils.get_multiview_dirs(args.normal, view_select=[view_i])[0]
            else:
                input_normal = input

            filename_normal = input_normal / configs.filename_normal
            if filename_normal.exists():
                normal_prior = utils.imread_normal(filename_normal)
                normal_prior[np.isnan(normal_prior)] = 0
                normal_prior[~mask_prior] = (0, 0, 1)
                normal_prior *= (1, -1, 1)  # flip y
            else:
                raise ValueError(f"Normal map {filename_normal} does not exist")
        else:
            normal_prior = np.zeros((height, width, 3), dtype=np.float32)

        # 2D Orientation
        orientation_prior = view.camera.projectLine(view.img_line)

        # Direction
        direction_world_prior = cv2.imread(str(input / configs.filename_direction), -1)  # xyz
        direction_prior = np.moveaxis(np.tensordot(R, direction_world_prior, axes=(1, -1)), 0, -1)  # world-to-local
        # direction_prior[..., 0] *= -1  # flip x

        # Align direction with orientation
        direction_on_image_plane = np.stack([np.cos(orientation_prior), -np.sin(orientation_prior), np.zeros_like(orientation_prior)], axis=-1)
        direction_prior[np.sum(direction_prior * direction_on_image_plane, axis=-1) < 0] *= -1

        if args.consistency is not None:
            filename_depth_consistency = mvs_consistency_subdirs[view_i] / configs.filename_consistency
            depth_consistensy = cv2.imread(str(filename_depth_consistency), -1)
        else:
            depth_consistensy = np.ones_like(depth_prior, dtype=np.float32)

        # Transfer data to device
        normal_prior = torch.Tensor(normal_prior).to(device)
        mask_prior = torch.Tensor(mask_prior).to(device)
        orientation_prior = torch.Tensor(orientation_prior).to(device)
        depth_prior = torch.Tensor(depth_prior).to(device)
        mask_prior = torch.Tensor(mask_prior).to(device)
        direction_prior = torch.Tensor(direction_prior).to(device)
        depth_consistensy = torch.Tensor(depth_consistensy).to(device)

        # Kernels for the partial derivatives
        kernel_u_posi = torch.Tensor([[0, 0, 0], [0, -1, 1], [0, 0, 0]]).to(device)
        kernel_u_nega = torch.Tensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]]).to(device)
        kernel_v_posi = torch.Tensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]]).to(device)
        kernel_v_nega = torch.Tensor([[0, 0, 0], [0, -1, 0], [0, 1, 0]]).to(device)
        kernels_u = torch.stack([kernel_u_posi, kernel_u_nega], dim=0)[:, None, :, :]
        kernels_v = torch.stack([kernel_v_posi, kernel_v_nega], dim=0)[:, None, :, :]

        cos = torch.cos(orientation_prior)
        sin = -torch.sin(orientation_prior)
        n_theta = cos * normal_prior[..., 0] + sin * normal_prior[..., 1]

        # PyTorch oprimizer and initialization
        z = torch.nn.Parameter(depth_prior.clone().detach())
        optimizer = torch.optim.Adam([z], lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay ** (1 / args.max_iter))
        masked_mean = WeightedMean(weights=mask_prior)

        # dtheta_z from 3D direction
        dir_x, dir_y, dir_z = direction_prior[..., 0], direction_prior[..., 1], direction_prior[..., 2]
        dtheta_z_from_dir = dir_z / torch.sqrt(dir_x**2 + dir_y**2)
        coff_dtheta_z = torch.sqrt(1 + dtheta_z_from_dir**2)

        output_sub = output / f"{view_i:02d}"
        output_sub.mkdir(parents=True, exist_ok=True)

        progress_bar_sub = trange(args.max_iter, position=1, leave=False)
        for iter in progress_bar_sub:
            optimizer.zero_grad()
            loss = torch.Tensor([0]).to(device)

            # Partial derivatives
            delta_u = z / fx
            delta_v = z / fy
            du_z_posi, du_z_nega = torch.nn.functional.conv2d(z[None, ...], kernels_u, padding="same") / delta_u
            dv_z_posi, dv_z_nega = torch.nn.functional.conv2d(z[None, ...], kernels_v, padding="same") / delta_v
            dtheta_z_posi = cos * du_z_posi + sin * dv_z_posi
            dtheta_z_nega = cos * du_z_nega + sin * dv_z_nega

            # Normal
            if args.w_normal > 0:
                img_normal_error_posi = (n_theta + normal_prior[..., 2] * dtheta_z_posi) ** 2
                img_normal_error_nega = (n_theta + normal_prior[..., 2] * dtheta_z_nega) ** 2
                img_normal_error = 0.5 * img_normal_error_posi + 0.5 * img_normal_error_nega
                loss_normal = args.w_normal * masked_mean(img_normal_error)
                loss += loss_normal
            else:
                loss_normal = torch.Tensor([0]).to(device)

            # Direction
            if args.w_direction > 0:
                img_direction_error_posi = (direction_prior[..., 2] - dtheta_z_posi / coff_dtheta_z) ** 2
                img_direction_error_nega = (direction_prior[..., 2] - dtheta_z_nega / coff_dtheta_z) ** 2
                img_direction_error = 0.5 * img_direction_error_posi + 0.5 * img_direction_error_nega
                loss_direction = args.w_direction * masked_mean(img_direction_error)
                loss += loss_direction
            else:
                loss_direction = torch.Tensor([0]).to(device)

            # Depth
            img_depth_error = (depth_prior - z) ** 2  # squared error
            img_depth_weighted_error = depth_consistensy * img_depth_error
            loss_depth = masked_mean(img_depth_weighted_error)
            loss += loss_depth

            loss.backward()
            optimizer.step()
            scheduler.step()

            desc = f"loss: {loss.item():.5f} depth: {loss_depth.item():.5f} normal: {loss_normal.item():.5f} direction: {loss_direction.item():.5f}"
            progress_bar_sub.set_description(desc)

            # Imshow
            if args.imshow and iter % 1000 == 0:
                z_np = z.cpu().detach().numpy()
                mask_np = mask_prior.cpu().detach().numpy() > 0

                vmin, vmax = np.percentile(z_np[mask_np], [1, 99])
                z_np_colored = utils.applyColorMap(z_np, "viridis_r", vmin=vmin, vmax=vmax)
                z_np_colored[~mask_np] = 255
                cv2.imshow("depth", z_np_colored)

                key = cv2.waitKey(30)
                if key == ord("q"):
                    break

        # End subloop
        z_np = z.cpu().detach().numpy()
        view.img_depth = z_np
        utils.write_singleview(output_sub, view, with_extra=True)


if __name__ == "__main__":
    main()
