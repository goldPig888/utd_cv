from _init_paths import *
from lib.Utils import *
from lib.XMemWrapper import XMemWrapper
from lib.SequenceLoader import SequenceLoader


def main():
    start_time = time.time()

    # Initialize XMemWrapper
    xmem_wrapper = XMemWrapper(model_type="XMem", device=device)

    # Load the sequence
    loader = SequenceLoader(sequence_folder, device=device)
    serials = loader.serials
    num_frames = loader.num_frames
    segmentation_folder = sequence_folder / "processed/segmentation"

    for serial in serials:
        logger.info(f"Processing serial {serial}...")

        # Read the first mask
        mask_files = sorted(
            (segmentation_folder / "init_segmentation" / serial).glob("mask_*.png")
        )

        if not mask_files:
            logger.warning(f"  - Initial mask does not exist! Skipping...")
            continue

        mask_0_file = mask_files[0]
        mask_0_frame_id = int(mask_0_file.stem.split("_")[1])
        mask_0 = read_mask_image(mask_0_file)

        save_folder = segmentation_folder / "xmem_segmentation" / serial
        save_folder.mkdir(parents=True, exist_ok=True)

        # Read the rgb images
        logger.info("  - Reading RGB images...")
        tqbar = tqdm(total=num_frames, ncols=60, colour="green")
        rgb_images = [None] * num_frames
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(loader.get_rgb_image, frame_id, serial): frame_id
                for frame_id in range(num_frames)
            }
            for future in concurrent.futures.as_completed(futures):
                rgb_images[futures[future]] = future.result()
                tqbar.update()
                tqbar.refresh()
        tqbar.close()

        logger.info("  - Running XMem segmentation...")
        xmem_wrapper.reset()

        tqbar = tqdm(total=num_frames, ncols=60, colour="green")
        mask_images = [None] * num_frames
        for frame_id in range(mask_0_frame_id, num_frames, 1):
            mask = xmem_wrapper.get_mask(
                rgb=rgb_images[frame_id],
                mask=mask_0 if frame_id == mask_0_frame_id else None,
            )
            mask_images[frame_id] = mask
            tqbar.update()
            tqbar.refresh()
        if mask_0_frame_id > 0:
            # xmem_wrapper.reset()
            for frame_id in range(mask_0_frame_id, -1, -1):
                mask = xmem_wrapper.get_mask(
                    rgb=rgb_images[frame_id],
                    mask=mask_0 if frame_id == mask_0_frame_id else None,
                    exhaustive=True,
                )
                mask_images[frame_id] = mask
                if frame_id != mask_0_frame_id:
                    tqbar.update()
                    tqbar.refresh()
        tqbar.close()

        logger.info("  - Saving XMem masks...")
        tqbar = tqdm(total=num_frames, ncols=60, colour="green")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(
                    write_mask_image,
                    save_folder / f"mask_{frame_id:06d}.png",
                    mask_images[frame_id],
                ): frame_id
                for frame_id in range(num_frames)
            }
            for future in concurrent.futures.as_completed(futures):
                future.result()
                tqbar.update()
                tqbar.refresh()
        tqbar.close()

        logger.info("  - Generating XMem vis images...")
        tqbar = tqdm(total=num_frames, ncols=60, colour="green")
        vis_images = [None] * num_frames
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(
                    draw_debug_image,
                    rgb_images[frame_id],
                    object_mask=mask_images[frame_id],
                    reduce_background=True,
                    alpha=0.7,
                ): frame_id
                for frame_id in range(num_frames)
            }
            for future in concurrent.futures.as_completed(futures):
                vis_images[futures[future]] = future.result()
                tqbar.update()
                tqbar.refresh()
        tqbar.close()

        del rgb_images
        del mask_images

        logger.info("  - Saving XMem vis images...")
        tqbar = tqdm(total=num_frames, ncols=60, colour="green")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(
                    write_rgb_image,
                    save_folder / f"vis_{frame_id:06d}.png",
                    vis_images[frame_id],
                ): frame_id
                for frame_id in range(num_frames)
            }
            for future in concurrent.futures.as_completed(futures):
                future.result()
                tqbar.update(1)
                tqbar.refresh()
            del futures
        tqbar.close()

        logger.info("  - Saving vis video...")
        create_video_from_rgb_images(
            video_path=segmentation_folder / "xmem_segmentation" / f"vis_{serial}.mp4",
            rgb_images=vis_images,
            fps=30,
        )
        del vis_images

    end_time = time.time()
    elapsed_time = end_time - start_time

    logger.info(f">>>>>>>>>> Done!!! ({elapsed_time:.2f} seconds)<<<<<<<<<<")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run X-Mem segmentation on a sequence."
    )
    parser.add_argument(
        "--sequence_folder", type=str, required=True, help="Path to the sequence folder"
    )
    args = parser.parse_args()
    sequence_folder = Path(args.sequence_folder).resolve()

    logger = get_logger("RunXMemSegmentation")

    if not torch.cuda.is_available():
        logger.error("CUDA is not available. Exiting...")
        exit()

    device = torch.device("cuda")

    main()
