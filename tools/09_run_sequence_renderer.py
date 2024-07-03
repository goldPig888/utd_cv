from _init_paths import *
from lib.Utils import *
from lib.SequenceLoader import SequenceLoader


def main():
    start_time = time.time()

    # Set the device to GPU if available, otherwise use CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize the SequenceLoader with the specified device
    loader = SequenceLoader(sequence_folder, load_mano=True, device=device)

    # Load necessary data from the loader
    num_frames = loader.num_frames
    cam_Ks = loader.Ks.cpu().numpy()
    cam_RTs = loader.extrinsics2world.cpu().numpy()
    mano_sides = loader.mano_sides

    # Load the object mesh
    obj_mesh = trimesh.load_mesh(loader.object_textured_mesh_file, process=False)

    # Load poses
    poses_o = np.load(poses_o_file)
    logger.info(f"poses_o loaded: {poses_o.shape}, {poses_o.dtype}")

    poses_m = np.load(poses_m_file)
    logger.info(f"poses_m loaded: {poses_m.shape}, {poses_m.dtype}")

    # Convert (quaterinion + translation) to 4x4 matrix
    poses_o = quat_to_mat(poses_o[0])

    # Convert MANO poses to torch tensor
    poses_m = np.concatenate(
        [poses_m[0 if side == "right" else 1] for side in mano_sides], axis=1
    )
    poses_m = torch.from_numpy(poses_m).to(device)

    # MANO mesh info
    mano_group_layer = loader.mano_group_layer
    mano_faces = loader.mano_faces
    mano_colors = loader.mano_colors
    mano_verts, _ = mano_group_layer(poses_m)
    mano_verts = mano_verts.cpu().numpy()

    # Render the sequence
    save_vis_folder = sequence_folder / "processed" / "sequence_rendering"
    save_vis_folder.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating rendered images...")
    vis_images = [None] * num_frames
    tqbar = tqdm(total=num_frames, ncols=80, colour="green")
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(
                runner_get_rendered_image,
                rgb_images=loader.get_rgb_image(frame_id),
                mano_mesh=trimesh.Trimesh(
                    vertices=mano_verts[frame_id],
                    faces=mano_faces,
                    vertex_colors=mano_colors,
                    process=False,
                ),
                object_mesh=obj_mesh,
                object_pose=poses_o[frame_id],
                cam_Ks=cam_Ks,
                cam_RTs=cam_RTs,
            ): frame_id
            for frame_id in range(len(poses_o))
        }
        for future in concurrent.futures.as_completed(futures):
            frame_id = futures[future]
            vis_images[frame_id] = future.result()
            tqbar.update(1)
            tqbar.refresh()
    tqbar.close()

    # Save the rendered images
    logger.info(f"Saving rendered images...")
    tqbar = tqdm(total=num_frames, ncols=80, colour="green")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(
                write_rgb_image,
                str(save_vis_folder / f"vis_{frame_id:06d}.png"),
                vis_images[frame_id],
            ): frame_id
            for frame_id in range(len(vis_images))
        }
        for future in concurrent.futures.as_completed(futures):
            future.result()
            tqbar.update(1)
            tqbar.refresh()
    tqbar.close()

    # Save the rendered video
    logger.info(f"Saving rendered video...")
    create_video_from_rgb_images(
        sequence_folder / f"vis_{sequence_folder.name}.mp4", vis_images, fps=30
    )

    end_time = time.time()
    elapsed_time = end_time - start_time

    logger.info(f">>>>>>>>>> Done!!! ({elapsed_time:.2f} seconds)<<<<<<<<<<")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Sequence Renderer")
    parser.add_argument(
        "--sequence_folder", type=str, required=True, help="Path to the sequence folder"
    )
    args = parser.parse_args()
    sequence_folder = Path(args.sequence_folder).resolve()

    logger = get_logger("SequenceRenderer")

    # Check if pose files exist
    poses_o_file = sequence_folder / "poses_o.npy"
    poses_m_file = sequence_folder / "poses_m.npy"
    if not poses_o_file.exists() or not poses_m_file.exists():
        logger.error(f"Pose files are missing. Exiting...")
        exit()

    main()
