import argparse
from _init_paths import *
from lib.Utils import *
from lib.SequenceLoader import SequenceLoader
from lib.XMemWrapper import XMemWrapper


def run_xmem_segmentation(xmem_wrapper, rgb_images, mask_0, save_folder):
    save_folder = Path(save_folder).resolve()
    save_folder.mkdir(parents=True, exist_ok=True)

    for i, rgb_image in tqdm(enumerate(rgb_images), total=len(rgb_images), ncols=60):
        if i == 0:
            mask = xmem_wrapper.get_mask(rgb_image, mask_0)
        else:
            mask = xmem_wrapper.get_mask(rgb_image)

        write_mask_image(save_folder / f"mask_{i:06d}.png", mask)

        # draw mask over image
        vis = draw_debug_image(
            rgb_image, object_mask=mask, reduce_background=True, alpha=0.7
        )
        write_rgb_image(save_folder / f"vis_{i:06d}.png", vis)


def main():
    loader = SequenceLoader(sequence_folder, device="cuda")
    serials = loader.serials
    num_frames = loader.num_frames

    xmem_wrapper = XMemWrapper(device=device)

    for serial in serials:
        tqdm.write(f"Processing serial {serial}...")
        # read the first mask
        mask_0_file = (
            sequence_folder
            / "processed/segmentation/init_segmentation"
            / serial
            / "mask_000000.png"
        )
        if not mask_0_file.exists():
            tqdm.write(f"  - Mask file {mask_0_file} does not exist! Exiting...")
            continue
        mask_0 = cv2.imread(str(mask_0_file), cv2.IMREAD_GRAYSCALE)

        # read the rgb images
        rgb_images = [None] * num_frames
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(loader.get_rgb_image, frame_id, serial): frame_id
                for frame_id in range(num_frames)
            }
            for future in concurrent.futures.as_completed(futures):
                rgb_images[futures[future]] = future.result()

        xmem_wrapper.reset()
        run_xmem_segmentation(
            xmem_wrapper,
            rgb_images,
            mask_0,
            sequence_folder / "processed/segmentation/xmem_segmentation" / serial,
        )


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
