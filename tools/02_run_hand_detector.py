import argparse

from _init_paths import *
from lib.Utils import *
from lib.HandDetector import HandDetector
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


def runner_mp_hand_detector(rgb_images, mp_config, idx=None):
    detector = HandDetector(mp_config)
    marks_result = np.full((len(rgb_images), 2, 21, 2), -1, dtype=np.int64)
    for frame_id in range(len(rgb_images)):
        hand_marks, hand_sides, hand_scores = detector.detect_one(rgb_images[frame_id])

        if hand_sides:
            # update hand sides if there are two same hand sides
            if len(hand_sides) == 2 and hand_sides[0] == hand_sides[1]:
                if hand_scores[0] >= hand_scores[1]:
                    hand_sides[1] = "right" if hand_sides[0] == "left" else "left"
                else:
                    hand_sides[0] = "right" if hand_sides[1] == "left" else "left"
            # update hand marks result
            for i, hand_side in enumerate(hand_sides):
                if hand_side == "right":
                    marks_result[frame_id][0] = hand_marks[i]
                if hand_side == "left":
                    marks_result[frame_id][1] = hand_marks[i]

    marks_result = marks_result.astype(np.int64)

    return marks_result, idx if idx is not None else marks_result


def main():
    device = MP_CONFIG["device"]
    logger = get_logger(log_level="DEBUG", log_name="HandDetector")

    loader = SequenceLoader(sequence_folder, device=device)
    serials = loader.serials
    num_frames = loader.num_frames
    mano_sides = loader.mano_sides

    logger.info(">>>>>>>>>> Running MediaPipe Hand Detection <<<<<<<<<<")

    logger.debug(
        f"""Config Settings:
    - Device: {device}
    - Serials: {serials}
    - Number of Frames: {num_frames}
    - Mano Sides: {mano_sides}"""
    )

    mp_handmarks = {serial: None for serial in serials}
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        tqbar = tqdm(total=len(serials), ncols=60, colour="green")
        workers = []
        for serial in serials:
            workers.append(
                executor.submit(
                    runner_mp_hand_detector,
                    rgb_images=[
                        loader.get_rgb_image(f_id, serial) for f_id in range(num_frames)
                    ],
                    mp_config=MP_CONFIG,
                    idx=serial,
                )
            )

        for worker in concurrent.futures.as_completed(workers):
            handmarks, serial = worker.result()
            mp_handmarks[serial] = handmarks
            tqbar.update()
            tqbar.refresh()
        tqbar.close()
        workers.clear()

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
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for serial in serials:
            tqdm.write(f"  - Saving hand detection results for {serial}")

            # Make folder to save hand detection results
            save_folder = sequence_folder / "processed" / "hand_detection" / serial
            make_clean_folder(save_folder)

            # Save hand detection results
            tqdm.write("    ** Saving Handmarks...")
            workers = []
            tqbar = tqdm(total=num_frames, ncols=60, colour="green")
            with concurrent.futures.ThreadPoolExecutor() as executor:
                for frame_id in range(num_frames):
                    workers.append(
                        executor.submit(
                            np.save,
                            save_folder / f"handmarks_{frame_id:06d}.npy",
                            mp_handmarks[serial][frame_id],
                        )
                    )
                for worker in concurrent.futures.as_completed(workers):
                    worker.result()
                    tqbar.update()
                    tqbar.refresh()
            tqbar.close()
            workers.clear()

            tqdm.write(f"    ** Saving vis images...")
            workers = []
            tqbar = tqdm(total=num_frames, ncols=60, colour="green")
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = {
                    executor.submit(
                        draw_debug_image,
                        loader.get_rgb_image(frame_id, serial),
                        hand_marks=mp_handmarks[serial][frame_id],
                        draw_boxes=True,
                        draw_hand_sides=True,
                        save_path=save_folder / f"vis_{frame_id:06d}.png",
                        return_image=False,
                    ): frame_id
                    for frame_id in range(num_frames)
                }
                for future in concurrent.futures.as_completed(futures):
                    future.result()
                    tqbar.update()
                    tqbar.refresh()
            tqbar.close()
            workers.clear()

    logger.info(">>>>>>>>>> Hand Detection Completed <<<<<<<<<<")


def args_parser():
    parser = argparse.ArgumentParser(description="Hand Detection")
    parser.add_argument(
        "--sequence_folder",
        type=str,
        required=True,
        help="Path to the sequence folder, python <python_file> --sequence_folder <path_to_sequence_folder>",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = args_parser()
    sequence_folder = Path(args.sequence_folder).resolve()

    main()
