import os
import cv2
import numpy as np
import mediapipe as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

def process_batch(detector, images, file_paths, output_directory):
    mp_drawing = mp.solutions.drawing_utils

    for image, img_path in zip(images, file_paths):
        try:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = detector.process(image_rgb)

            if results.multi_hand_landmarks:
                handmarks = []
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.append([lm.x * image.shape[1], lm.y * image.shape[0]])
                    handmarks.append(landmarks)
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

                handmarks = np.array(handmarks)
                folder_name = os.path.basename(os.path.dirname(img_path))
                frame_number = os.path.splitext(os.path.basename(img_path))[0].split('_')[-1]

                handmarks_output_path = os.path.join(output_directory, folder_name, f"handmarks_{frame_number}.npy")
                vis_output_path = os.path.join(output_directory, folder_name, f"vis_{frame_number}.png")

                os.makedirs(os.path.dirname(handmarks_output_path), exist_ok=True)

                np.save(handmarks_output_path, handmarks)
                print(f"Handmarks saved to {handmarks_output_path}")

                cv2.imwrite(vis_output_path, image)
                print(f"Visualization saved to {vis_output_path}")
            else:
                print(f"No hand landmarks detected in {img_path}")
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")

class HandDetector:
    def __init__(self, input_directory, output_directory):
        self.input_directory = input_directory
        self.output_directory = output_directory
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

    def detect_handmarks(self):
        for root, _, files in os.walk(self.input_directory):
            print(f"Processing directory: {root}")
            batch_images = []
            batch_file_paths = []

            for file in sorted(files):
                if file.endswith(".jpg"):
                    img_path = os.path.join(root, file)
                    image = cv2.imread(img_path)
                    if image is not None:
                        batch_images.append(image)
                        batch_file_paths.append(img_path)

            if batch_images:
                process_batch(self.hands, batch_images, batch_file_paths, self.output_directory)

def process_camera(input_directory, output_directory):
    detector = HandDetector(input_directory, output_directory)
    detector.detect_handmarks()

def process_folder(folder_dir, output_directory):
    print(f"Processing folder: {folder_dir}")
    hand_detection_output_dir = os.path.join(folder_dir, "processed", "hand_detection")
    os.makedirs(hand_detection_output_dir, exist_ok=True)

    process_camera(folder_dir, hand_detection_output_dir)

def process_bag(bag_dir, output_directory):
    try:
        for folder in os.listdir(bag_dir):
            folder_path = os.path.join(bag_dir, folder)
            if os.path.isdir(folder_path):
                process_folder(folder_path, output_directory)
    except Exception as e:
        print(f"Error processing bag directory {bag_dir}: {e}")

def process_bags(base_directory, output_directory):
    bag_dirs = [os.path.join(base_directory, bag_dir) for bag_dir in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, bag_dir))]
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(process_bag, bag_dir, output_directory) for bag_dir in bag_dirs]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error in concurrent execution: {e}")

def main():
    base_directory = "/home/jikaiwang/GitHub/summer_camp/data/recordings"
    output_directory = base_directory
    process_bags(base_directory, output_directory)
    print("done")

if __name__ == '__main__':
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()
    print(f"Total time: {end_time - start_time} seconds")
