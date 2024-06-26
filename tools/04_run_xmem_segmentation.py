import argparse
from _init_paths import *
from lib.Utils import *
from lib.SequenceLoader import SequenceLoader
from lib.XMemWrapper import XMemWrapper


def main():
    loader = SequenceLoader(sequence_folder, device=device)
    serials = loader.serials
    num_frames = loader.num_frames
    xmem_wrapper = XMemWrapper(model_type="XMem", device=device)

    for serial in serials:
        tqdm.write(f"Processing serial {serial}...")
        # read the first mask
        mask_0_file = sorted(
            (
                sequence_folder / "processed/segmentation/init_segmentation" / serial
            ).glob("mask_*.png")
        )[0]
        if not mask_0_file.exists():
            tqdm.write(f"  - Mask file {mask_0_file} does not exist! Exiting...")
            continue

        mask_0_frame_id = int(mask_0_file.stem.split("_")[1])
        mask_0 = read_mask_image(mask_0_file)

        save_folder = (
            sequence_folder / "processed/segmentation/xmem_segmentation" / serial
        )
        save_folder.mkdir(parents=True, exist_ok=True)

        # read the rgb images
        tqdm.write("  - Reading RGB images...")
        tqbar = tqdm(total=num_frames, ncols=60, colour="green")
        rgb_images = [None] * num_frames
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(loader.get_rgb_image, frame_id, serial): frame_id
                for frame_id in range(num_frames)
            }
            for future in concurrent.futures.as_completed(futures):
                rgb_images[futures[future]] = future.result()
                tqbar.update(1)
                tqbar.refresh()
            del futures
        tqbar.close()

        tqdm.write("  - Running XMem segmentation...")
        xmem_wrapper.reset()

        tqbar = tqdm(total=num_frames, ncols=60, colour="green")
        mask_images = [None] * num_frames
        for frame_id in range(mask_0_frame_id, num_frames, 1):
            mask = xmem_wrapper.get_mask(
                rgb=rgb_images[frame_id],
                mask=mask_0 if frame_id == mask_0_frame_id else None,
            )
            mask_images[frame_id] = mask
            tqbar.update(1)
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
                    tqbar.update(1)
                    tqbar.refresh()
        tqbar.close()

        tqdm.write("  - Saving XMem masks...")
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
                tqbar.update(1)
                tqbar.refresh()
            del futures
        tqbar.close()

        tqdm.write("  - Generating XMem vis images...")
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
                tqbar.update(1)
                tqbar.refresh()
            del futures
        tqbar.close()

        del rgb_images, mask_images

        tqdm.write("  - Saving XMem vis images...")
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

        tqdm.write("  - Saving vis video...")
        create_video_from_rgb_images(
            video_path=sequence_folder
            / "processed/segmentation/xmem_segmentation"
            / f"vis_{serial}.mp4",
            images=vis_images,
            fps=30,
        )
        del vis_images

    del loader, xmem_wrapper

    tqdm.write(">>>>>>>>>> Done. <<<<<<<<<<<<")


def args_parser():
    parser = argparse.ArgumentParser(
        description="Run X-Mem segmentation on a sequence."
    )
    parser.add_argument(
        "--sequence_folder",
        type=str,
        help="Path to the folder containing the sequence.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    sequence_folder = Path(args_parser().sequence_folder).resolve()

    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting...")
        exit()
    device = torch.device("cuda")

    main()
