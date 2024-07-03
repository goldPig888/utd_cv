from _init_paths import *
from lib.Utils import *
from lib.Layers import MANOLayer
from lib.SequenceLoader import SequenceLoader

# Set the start method to 'spawn'
try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass


class ManoSolver:
    def __init__(self, sequence_folder, device="cpu") -> None:
        self._save_folder = Path(sequence_folder) / "processed" / "hand_detection"
        self._device = device

        self._logger = get_logger("ManoSolver")

        # load parameters from data loader
        self._data_loader = SequenceLoader(sequence_folder, device=device)
        self._serials = self._data_loader.serials
        self._num_frames = self._data_loader.num_frames
        self._mano_sides = self._data_loader.mano_sides
        self._intrinsics = self._data_loader.Ks.cpu().numpy()
        self._extrinsics = self._data_loader.extrinsics2world.cpu().numpy()
        self._mano_beta = self._data_loader.mano_beta.cpu().numpy()

        # load 3D joints
        self._load_joints_3d()

        self._mse_loss = torch.nn.MSELoss(reduction="sum").to(self._device)
        self._iter_solve = 21
        self._iter_poses = [2000] + [20] * (self._iter_solve - 1)
        self._iter_betas = [100] + [20] * (self._iter_solve - 1)

    def run(self):
        start_time = time.time()

        poses_opt = []
        betas_opt = []
        verts_m = []
        faces_m = []
        colors_m = []

        for i, side in enumerate(self._mano_sides):
            target_joints_3d = self._joints_3d[0 if side == "right" else 1]
            if torch.any(target_joints_3d == -1):
                self._logger.warning(
                    "No valid 3D joints for side: {}, skipping...".format(side)
                )
                continue

            pose_opt, beta_opt, verts, faces, colors = self.solve(
                side,
                target_joints_3d,
            )
            poses_opt.append(pose_opt)
            betas_opt.append(beta_opt)
            verts_m.append(verts)
            faces_m.append(faces)
            colors_m.append(colors)

            # Update MANO beta parameters
            self._mano_beta = beta_opt.reshape(-1)
            self._iter_betas = [0] * self._iter_solve

        if len(poses_opt) == 0:
            self._logger.warning("No valid side to solve, exiting...")
            return

        # Save the results
        poses_opt = np.stack(poses_opt, axis=0).astype(np.float32)
        betas_opt = np.stack(betas_opt, axis=0).astype(np.float32)
        np.savez_compressed(
            str(self._save_folder / "hamer_mano_solver_results.npz"),
            poses=poses_opt,
            betas=betas_opt,
        )

        # Visualize the results
        for i in range(len(faces_m)):
            faces_m[i] += i * NUM_MANO_VERTS
        faces_m = np.vstack(faces_m)
        colors_m = np.vstack(colors_m)
        verts_m = np.stack(verts_m, axis=1).reshape(self._num_frames, -1, 3)
        print(
            f"verts_m: {verts_m.shape}, faces_m: {faces_m.shape}, colors_m: {colors_m.shape}"
        )

        self._logger.info("Generating vis images...")
        vis_images = [None] * self._num_frames
        tqbar = tqdm(total=self._num_frames, ncols=80, colour="green")
        with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
            futures = {
                executor.submit(
                    runner_get_rendered_image,
                    rgb_images=self._data_loader.get_rgb_image(frame_id),
                    mano_mesh=trimesh.Trimesh(
                        vertices=verts_m[frame_id],
                        faces=faces_m,
                        vertex_colors=colors_m,
                        process=False,
                    ),
                    cam_Ks=self._intrinsics,
                    cam_RTs=self._extrinsics,
                ): frame_id
                for frame_id in range(self._num_frames)
            }
            for future in concurrent.futures.as_completed(futures):
                vis_images[futures[future]] = future.result()
                tqbar.update()
                tqbar.refresh()
        tqbar.close()

        self._logger.info("Saving vis images...")
        save_vis_folder = self._save_folder / "vis" / "hamer_mano_solver"
        save_vis_folder.mkdir(parents=True, exist_ok=True)
        tqbar = tqdm(total=self._num_frames, ncols=80, colour="green")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(
                    write_rgb_image,
                    save_vis_folder / f"vis_{frame_id:06d}.png",
                    vis_images[frame_id],
                ): frame_id
                for frame_id in range(self._num_frames)
            }
            for future in concurrent.futures.as_completed(futures):
                future.result()
                tqbar.update()
                tqbar.refresh()
        tqbar.close()

        self._logger.info("Saving vis video...")
        create_video_from_rgb_images(
            self._save_folder / "vis" / f"{save_vis_folder.name}.mp4",
            vis_images,
            fps=30,
        )

        end_time = time.time()
        elapsed_time = end_time - start_time

        self._logger.info(f">>>>>>>>>> Done!!! ({elapsed_time:.2f} seconds) <<<<<<<<<<")

    def _load_joints_3d(self):
        joints_3d = np.load(self._save_folder / "hamer_hand_joints_3d_interpolated.npy")
        self._joints_3d = torch.from_numpy(joints_3d).to(self._device)

    def mano_layer_forward(self, mano_layer, pose, betas):
        # Forward pass through the MANO layer
        verts, joints = mano_layer._mano_layer(pose[:, :48], betas, pose[:, 48:])

        # Ensure contiguous memory layout
        verts = verts.contiguous()
        joints = joints.contiguous()

        # If batch size is 1, remove the batch dimension
        if pose.size(0) == 1:
            verts = verts.squeeze(0)
            joints = joints.squeeze(0)

        # Convert units from millimeters to meters
        verts /= 1000.0
        joints /= 1000.0

        return verts, joints

    def loss_mse(self, x):
        """
        Calculates the mean squared error (MSE) loss for the given tensor `x` against a tensor of zeros.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Computed MSE loss.
        """
        target = torch.zeros_like(x, device=self._device)
        loss = self._mse_loss(x, target)
        loss /= x.size(0)  # Divide by the batch size to get the mean loss per sample
        return loss

    def save_plot_log_loss(self, save_path, log_loss) -> None:
        """
        Plot the log of losses recorded during optimization.

        Parameters:
        - log_loss (np.ndarray): Array containing the logged losses.
        Shape should be (num_iterations, 4), where the columns represent:
        - Total loss
        - Joints loss
        - Poses loss
        - Betas loss
        """
        num_iterations = log_loss.shape[0]
        iterations = np.arange(1, num_iterations + 1)

        # Extract individual losses
        total_loss = log_loss[:, 0]
        joints_loss = log_loss[:, 1]
        poses_loss = log_loss[:, 2]
        betas_loss = log_loss[:, 3]

        # Plot the losses
        plt.figure(figsize=(12, 8))
        plt.plot(iterations, total_loss, label="Total Loss", color="blue")
        plt.plot(iterations, joints_loss, label="Joints Loss", color="green")
        plt.plot(iterations, poses_loss, label="Poses Loss", color="red")
        plt.plot(iterations, betas_loss, label="Betas Loss", color="orange")

        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("Optimization Losses Over Iterations")
        plt.legend()
        plt.grid(True)

        plt.savefig(str(save_path))

    def solve(self, mano_side, target_joints_3d):
        """
        Solves for MANO parameters given the target 3D joint positions.

        Args:
            mano_side (str): 'right' or 'left' indicating the hand side.
            target_joints_3d (torch.Tensor): Target 3D joint positions.

        Returns:
            tuple: Optimized pose, beta, vertex positions, faces, and colors.
        """
        self._logger.info(f"Solving MANO for side: {mano_side}")

        # Initialize the MANO layer
        mano_layer = MANOLayer(mano_side, self._mano_beta).to(self._device)

        # Initialize the parameters to optimize
        poses = torch.zeros(
            (target_joints_3d.size(0), 51),
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        betas = torch.zeros(
            (1, 10), dtype=torch.float32, device=device, requires_grad=True
        )

        # Assign the MANO pose transformation
        poses.data[:, -3:] = target_joints_3d[:, 0].clone()

        # Initialize betas with the provided MANO beta parameters
        betas.data[:] = torch.from_numpy(self._mano_beta).to(self._device).clone()

        # Create an optimizer for the poses and betas
        optimizer = torch.optim.Adam([poses, betas], lr=0.01)

        log_loss = []
        log_pose = []
        log_beta = []

        for j in range(self._iter_solve):
            for i in range(self._iter_poses[j] + self._iter_betas[j]):
                s = time.time()
                optimizer.zero_grad()

                # Forward pass
                _, joints = self.mano_layer_forward(
                    mano_layer, poses, betas.expand(poses.size(0), -1)
                )

                # Calculate losses
                loss_joints = self.loss_mse(joints - target_joints_3d) * 1e3
                loss_betas = self.loss_mse(betas)
                loss_poses = self.loss_mse(poses[:, 3:48])
                loss = 1.0 * loss_joints + 0.01 * loss_poses + 0.01 * loss_betas

                # Backward pass and optimization step
                loss.backward()
                if i < self._iter_poses[j]:
                    betas.grad = None
                else:
                    poses.grad = None
                optimizer.step()

                # Log the losses and parameters
                log_loss.append(
                    torch.stack([loss, loss_joints, loss_poses, loss_betas])
                    .detach()
                    .cpu()
                    .clone()
                )
                log_pose.append(poses.detach().cpu().clone())
                log_beta.append(betas.detach().cpu().clone())

                e = time.time()
                current_iter = (
                    sum(self._iter_poses[:j]) + sum(self._iter_betas[:j]) + i + 1
                )
                if current_iter % 100 == 0:
                    print(
                        f"iter: {current_iter:06d} | loss: {loss.item():11.8f} | "
                        f"joints: {loss_joints.item():11.8f} | poses: {loss_poses.item():11.8f} | "
                        f"betas: {loss_betas.item():11.8f} | time: {e - s:11.3f}"
                    )

        log_loss = torch.stack(log_loss).numpy()
        log_pose = torch.stack(log_pose).numpy()
        log_beta = torch.stack(log_beta).numpy()

        data = {
            "log_beta": log_beta,
            "log_pose": log_pose,
            "log_loss": log_loss,
        }

        # Save the plot of the losses
        self.save_plot_log_loss(
            self._save_folder / f"hamer_mano_solver_loss_{mano_side}.png", log_loss
        )

        # Save the results
        np.savez(str(self._save_folder / f"hamer_mano_solver_{mano_side}.npz"), **data)

        # Extract the optimized poses and betas
        pose_opt = log_pose[-1]
        beta_opt = log_beta[-1]

        # Forward pass to get the vertex positions with optimized parameters
        verts_m, _ = self.mano_layer_forward(
            mano_layer, poses, betas.expand(poses.size(0), -1)
        )
        verts_m = verts_m.detach().cpu().numpy()

        # Extract faces and colors from the MANO layer
        new_faces = np.array(
            NEW_MANO_FACES_RIGHT if mano_side == "right" else NEW_MANO_FACES_LEFT
        )
        faces_m = np.concatenate([mano_layer.f.cpu().numpy(), new_faces], axis=0)
        hand_color = (
            HAND_COLORS[1].rgb_norm if mano_side == "right" else HAND_COLORS[2].rgb_norm
        )
        colors_m = np.tile(hand_color, (NUM_MANO_VERTS, 1)).astype(np.float32)

        return pose_opt, beta_opt, verts_m, faces_m, colors_m


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HaMeR MANO Pose & Shape Solver")
    parser.add_argument(
        "--sequence_folder", type=str, required=True, help="Path to the sequence folder"
    )
    args = parser.parse_args()
    sequence_folder = Path(args.sequence_folder).resolve()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    mano_solver = ManoSolver(sequence_folder, device)
    mano_solver.run()
