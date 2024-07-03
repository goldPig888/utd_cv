from _init_paths import *
from lib.Utils import *
from lib.MPHandDetector import MPHandDetector
from lib.SequenceLoader import SequenceLoader


MP_CONFIG = {
    "max_num_hands": 2,
    "min_hand_detection_confidence": 0.1,
    "min_tracking_confidence": 0.5,
    "min_hand_presence_confidence": 0.5,
    "running_mode": "video",
    "frame_rate": 30,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


def runner_draw_handmarks_results(rgb_images, handmarks, serials):
    """
    Draws hand landmarks on a list of RGB images and displays them with their respective serials.

    Args:
        rgb_images (list of np.ndarray): List of RGB images.
        handmarks (list of list): List of hand landmarks for each image.
        serials (list of str): List of serial names for each image.

    Returns:
        np.ndarray: Array of images with drawn hand landmarks and other debug information.
    """
    if not (len(rgb_images) == len(handmarks) == len(serials)):
        raise ValueError(
            "The length of rgb_images, handmarks, and serials must be the same."
        )

    vis_images = [
        draw_debug_image(
            rgb_image,
            hand_marks=handmarks[idx],
            draw_boxes=True,
            draw_hand_sides=True,
        )
        for idx, rgb_image in enumerate(rgb_images)
    ]

    vis_images = display_images(images=vis_images, names=serials, return_array=True)
    return vis_images


def runner_mp_hand_detector(rgb_images, mp_config):
    """
    Runs hand detection on a series of RGB images using the provided MediaPipe configuration.

    Args:
        rgb_images (list of np.ndarray): List of RGB images.
        mp_config (dict): Configuration dictionary for the MediaPipe hand detector.

    Returns:
        np.ndarray: Array of hand marks with shape (num_frames, 2, 21, 2).
                    The marks are filled with -1 where hands are not detected.
    """
    detector = MPHandDetector(mp_config)
    num_frames = len(rgb_images)
    marks_result = np.full((num_frames, 2, 21, 2), -1, dtype=np.int64)

    for frame_id in range(num_frames):
        hand_marks, hand_sides, hand_scores = detector.detect_one(rgb_images[frame_id])

        if hand_sides:
            # Ensure there are no two same hand sides
            if len(hand_sides) == 2 and hand_sides[0] == hand_sides[1]:
                if hand_scores[0] >= hand_scores[1]:
                    hand_sides[1] = "right" if hand_sides[0] == "left" else "left"
                else:
                    hand_sides[0] = "right" if hand_sides[1] == "left" else "left"

            # Update hand marks result
            for i, hand_side in enumerate(hand_sides):
                side_index = 0 if hand_side == "right" else 1
                marks_result[frame_id][side_index] = hand_marks[i]

    return marks_result.astype(np.int64)


def main():
    start_time = time.time()  # Record the start time

    device = MP_CONFIG["device"]
    logger = get_logger(log_level="DEBUG", log_name="MPHandDetector")

    loader = SequenceLoader(sequence_folder, device=device)
    serials = loader.serials
    num_frames = loader.num_frames
    mano_sides = loader.mano_sides
    MP_CONFIG["max_num_hands"] = len(mano_sides)

    logger.info(">>>>>>>>>> Running MediaPipe Hand Detection <<<<<<<<<<")

    logger.debug(
        f"""Config Settings:
    - Device: {device}
    - Serials: {serials}
    - Number of Frames: {num_frames}
    - Mano Sides: {mano_sides}"""
    )

    # Initialize results dictionary
    mp_handmarks = {serial: None for serial in serials}

    # Process each serial using multiprocessing
    tqbar = tqdm(total=len(serials), ncols=60, colour="green")
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(
                runner_mp_hand_detector,
                rgb_images=[
                    loader.get_rgb_image(f_id, serial) for f_id in range(num_frames)
                ],
                mp_config=MP_CONFIG,
            ): serial
            for serial in serials
        }
        for future in concurrent.futures.as_completed(futures):
            mp_handmarks[futures[future]] = future.result()
            tqbar.update()
            tqbar.refresh()
    tqbar.close()

    logger.info("*** Updating Hand Detection Results with 'mano_sides' ***")
    if mano_sides is not None and len(mano_sides) == 1:
        for serial in serials:
            for frame_id in range(num_frames):
                if "right" in mano_sides:
                    if np.any(mp_handmarks[serial][frame_id][0] == -1) and np.all(
                        mp_handmarks[serial][frame_id][1] != -1
                    ):
                        mp_handmarks[serial][frame_id][0] = mp_handmarks[serial][
                            frame_id
                        ][1]
                    mp_handmarks[serial][frame_id][1] = -1
                if "left" in mano_sides:
                    if np.any(mp_handmarks[serial][frame_id][1] == -1) and np.all(
                        mp_handmarks[serial][frame_id][0] != -1
                    ):
                        mp_handmarks[serial][frame_id][1] = mp_handmarks[serial][
                            frame_id
                        ][0]
                    mp_handmarks[serial][frame_id][0] = -1

    logger.info("*** Saving Hand Detection Results ***")
    save_folder = sequence_folder / "processed" / "hand_detection"
    save_folder.mkdir(parents=True, exist_ok=True)

    # swap axis to (2, num_frames, 21, 2)
    for serial in serials:
        mp_handmarks[serial] = np.swapaxes(mp_handmarks[serial], 0, 1).astype(np.int64)

    np.savez_compressed(save_folder / "mp_handmarks_results.npz", **mp_handmarks)

    logger.info("*** Generating Hand Detection Visualizations ***")
    tqbar = tqdm(total=num_frames, ncols=60, colour="green")
    vis_images = [None] * num_frames
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(
                runner_draw_handmarks_results,
                loader.get_rgb_image(frame_id),
                [mp_handmarks[serial][:, frame_id] for serial in serials],
                serials,
            ): frame_id
            for frame_id in range(num_frames)
        }
        for future in concurrent.futures.as_completed(futures):
            vis_images[futures[future]] = future.result()
            tqbar.update()
            tqbar.refresh()
    tqbar.close()

    logger.info("*** Saving Visualization Images ***")
    save_vis_folder = save_folder / "vis" / "mp_handmarks"
    save_vis_folder.mkdir(parents=True, exist_ok=True)

    tqbar = tqdm(total=num_frames, ncols=60, colour="green")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(
                write_rgb_image,
                save_vis_folder / f"vis_{frame_id:06d}.png",
                vis_images[frame_id],
            ): frame_id
            for frame_id in range(num_frames)
        }
        for future in concurrent.futures.as_completed(futures):
            future.result()
            tqbar.update()
            tqbar.refresh()
    tqbar.close()

    logger.info("*** Creating Visualization Video ***")
    create_video_from_rgb_images(
        save_folder / "vis" / "mp_handmarks.mp4", vis_images, fps=30
    )

    end_time = time.time()  # Record the end time
    elapsed_time = end_time - start_time  # Calculate the elapsed time

    logger.info(f">>>>>>>>>> Done!!! ({elapsed_time:.2f} seconds) <<<<<<<<<<")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MediaPipe Hand Detection")
    parser.add_argument(
        "--sequence_folder", type=str, required=True, help="Path to the sequence folder"
    )
    args = parser.parse_args()
    sequence_folder = Path(args.sequence_folder).resolve()

    main()
