import os
import cv2
import numpy as np
import mediapipe as mp

class HandDetector:
    def __init__(self, input_directory, output_directory):
        self.input_directory = input_directory
        self.output_directory = output_directory
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils

    def detect_handmarks(self):
        for root, _, files in os.walk(self.input_directory):
            for file in files:
                if file.endswith(".jpg"):
                    img_path = os.path.join(root, file)
                    self.process_image(img_path, root)

    def process_image(self, img_path, root):
        image = cv2.imread(img_path)
        results = self.hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:
            handmarks = []
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.append([lm.x * image.shape[1], lm.y * image.shape[0]])
                handmarks.append(landmarks)
                self.mp_drawing.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

            handmarks = np.array(handmarks)
            serial_number = os.path.basename(root)
            frame_number = os.path.splitext(os.path.basename(img_path))[0].split('_')[-1]

            handmarks_output_path = os.path.join(self.output_directory, serial_number, f"handmarks_{frame_number}.npy")
            vis_output_path = os.path.join(self.output_directory, serial_number, f"vis_{frame_number}.png")

            os.makedirs(os.path.dirname(handmarks_output_path), exist_ok=True)

            np.save(handmarks_output_path, handmarks)
            cv2.imwrite(vis_output_path, image)

            print(f"Saved handmarks to {handmarks_output_path}")
            print(f"Saved visualization to {vis_output_path}")
        else:
            print(f"No hand landmarks detected in {img_path}")
