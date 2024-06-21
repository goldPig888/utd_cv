import os
import cv2
import numpy as np
import mediapipe as mp
import dask
import dask.delayed
import torch
import time
import aiofiles
import asyncio
from dask.distributed import Client, LocalCluster

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

async def async_read_image(img_path):
    async with aiofiles.open(img_path, mode='rb') as f:
        image = np.frombuffer(await f.read(), dtype=np.uint8)
        return cv2.imdecode(image, cv2.IMREAD_COLOR)

@dask.delayed
def process_image_batch(img_paths, output_directory, draw_landmarks=False):
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.1)

    async def process_images():
        tasks = [async_read_image(img_path) for img_path in img_paths]
        images = await asyncio.gather(*tasks)
        return images

    try:
        images = asyncio.run(process_images())
        for img_path, image in zip(img_paths, images):
            if image is None:
                print(f"Image not found or cannot be read: {img_path}")
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            if results.multi_hand_landmarks:
                handmarks = []
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    landmarks = [[int(lm.x * image.shape[1]), int(lm.y * image.shape[0])] for lm in hand_landmarks.landmark]
                    handmarks.append(landmarks)

                    if draw_landmarks:
                        mp_drawing.draw_landmarks(
                            image,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style())

                if results.multi_handedness:
                    hand_labels = [(results.multi_handedness[idx].classification[0].label,
                                    results.multi_handedness[idx].classification[0].score)
                                    for idx in range(len(results.multi_handedness))]

                    left_hands = [label for label, _ in hand_labels if label == "Left"]
                    if len(left_hands) == 2:
                        left_confidences = [score for label, score in hand_labels if label == "Left"]
                        if left_confidences[0] > left_confidences[1]:
                            hand_labels[0] = ("Right", left_confidences[1])
                        else:
                            hand_labels[1] = ("Right", left_confidences[0])

                    for idx, (label, score) in enumerate(hand_labels):
                        text = f'{label} ({score:.2f})'
                        coords = (handmarks[idx][0][0], handmarks[idx][0][1] - 10)
                        cv2.putText(image, text, coords, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                handmarks = np.array(handmarks)
                folder_name = os.path.basename(os.path.dirname(img_path))
                frame_number = os.path.splitext(os.path.basename(img_path))[0].split('_')[-1]

                handmarks_output_path = os.path.join(output_directory, "processed", "hand_detection", folder_name, f"handmarks_{frame_number}.npy")
                vis_output_path = os.path.join(output_directory, "processed", "hand_detection", folder_name, f"vis_{frame_number}.png")

                os.makedirs(os.path.dirname(handmarks_output_path), exist_ok=True)

                np.save(handmarks_output_path, handmarks)
                if draw_landmarks:
                    cv2.imwrite(vis_output_path, image)
            else:
                print(f"No hand landmarks detected in {img_path}")
    except Exception as e:
        print(f"Error processing image batch: {e}")
    finally:
        hands.close()

def process_images_in_batches(file_paths, output_directory, batch_size=10, draw_landmarks=False):
    batched_file_paths = [file_paths[i:i + batch_size] for i in range(0, len(file_paths), batch_size)]
    tasks = [process_image_batch(batch, output_directory, draw_landmarks) for batch in batched_file_paths]
    dask.compute(*tasks, scheduler='threads')

class HandDetector:
    def __init__(self, input_directory, output_directory, batch_size=10, draw_landmarks=False):
        self.input_directory = input_directory
        self.output_directory = output_directory
        self.batch_size = batch_size
        self.draw_landmarks = draw_landmarks

    def detect_handmarks(self):
        batch_file_paths = []
        for root, _, files in os.walk(self.input_directory):
            print(f"Processing directory: {root}")
            batch_file_paths.extend([os.path.join(root, file) for file in sorted(files) if file.endswith(".jpg")])

        process_images_in_batches(batch_file_paths, self.output_directory, self.batch_size, self.draw_landmarks)

def process_bag(bag_dir, output_directory, batch_size=10, draw_landmarks=False):
    detector = HandDetector(bag_dir, output_directory, batch_size, draw_landmarks)
    detector.detect_handmarks()

def process_bags(base_directory, output_directory, batch_size=10, draw_landmarks=False):
    cluster = LocalCluster(n_workers=8, threads_per_worker=2)  # Adjust based on your machine's capability
    client = Client(cluster)
    bag_dirs = [os.path.join(base_directory, bag_dir) for bag_dir in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, bag_dir))]
    tasks = [dask.delayed(process_bag)(bag_dir, output_directory, batch_size, draw_landmarks) for bag_dir in bag_dirs]
    dask.compute(*tasks, scheduler='distributed')
    client.close()

def main():
    base_directory = "/Users/mayespinola/Documents/GitHub/utd_cv/data/recordings"
    output_directory = base_directory
    process_bags(base_directory, output_directory, batch_size=10, draw_landmarks=True)
    print("done")

if __name__ == '__main__':
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()
    print(f"Total time: {end_time - start_time} seconds")
