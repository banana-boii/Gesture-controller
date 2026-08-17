import cv2
import mediapipe as mp
import csv
import os
from typing import Any

model_path = "hand_landmarker.task"

cam = cv2.VideoCapture(0)
print("isOpened:", cam.isOpened())
print("width/height:", cam.get(cv2.CAP_PROP_FRAME_WIDTH), cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

frame_Width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_Height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

"""
def print_result(result: Any, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result
"""

options = HandLandmarkerOptions(
    base_options = BaseOptions(model_asset_path = model_path),
    running_mode = VisionRunningMode.VIDEO,
    #result_callback = print_result,
    num_hands = 2
)

csv_filename = "gesture_dataset.csv"
file_exists = os.path.isfile(csv_filename)

Gesture_Map = {
    ord('1'): 0,
    ord('2'): 1,
    ord('3'): 2,
    ord('4'): 3
}

#latest_result = None
with HandLandmarker.create_from_options(options) as landmarker:
    with open(csv_filename, mode='a', newline='') as f:
        writer = csv.writer(f)

        if not file_exists:
            header = ['label']
            for i in range(21):
                header.extend([f'x{i}', f'y{i}', f'z{i}'])
            writer.writerow(header)
        print('Starting')

        while True:

            ret, frame = cam.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

            hand_landmarker_result = landmarker.detect_for_video(mp_image, int(cv2.getTickCount() / cv2.getTickFrequency() * 1000))

            normalized_coords = []
            if hand_landmarker_result.hand_landmarks:
                hand_landmarks = hand_landmarker_result.hand_landmarks[0]
                wrist = hand_landmarks[0]
                wrist_x, wrist_y, wrist_z = wrist.x, wrist.y, wrist.z

                for lm in hand_landmarks:
                    normalized_coords.extend([
                        lm.x - wrist_x,
                        lm.y - wrist_y,
                        lm.z - wrist_z
                    ])
                cx, cy = int(lm.x * frame_Width), int(lm.y * frame_Height)
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

            cv2.imshow("Hand Landmarker", frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            if key in Gesture_Map and normalized_coords:
                label = Gesture_Map[key]
                writer.writerow([label] + normalized_coords)
                print(f"Logged gesture {label}")


cam.release()
cv2.destroyAllWindows()