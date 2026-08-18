import cv2
import mediapipe as mp
import csv
import os
from typing import Any
from collections import deque

model_path = "hand_landmarker.task"
WINDOW_SIZE = 10

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
    ord('1'): 'closed fist',
    ord('2'): 'open palm',
    ord('3'): 'thumbs up',
    ord('4'): 'thumbs down',
    ord('5'): 'pointing left',
    ord('6'): 'pointing right',
    ord('7'): 'okay sign',
    ord('8'): 'swip left',
    ord('9'): 'swip right',
    ord(' '): 'idle/normal'
}

frame_buffer = deque(maxlen = WINDOW_SIZE)
n = 1

#latest_result = None
with HandLandmarker.create_from_options(options) as landmarker:
    with open(csv_filename, mode='a', newline='') as f:
        writer = csv.writer(f)

        if not file_exists:
            header = ['label']
            for frame_idx in range(WINDOW_SIZE):
                for i in range(21):
                    header.extend([f'x{i}_f{frame_idx}', f'y{i}_f{frame_idx}', f'z{i}_f{frame_idx}'])
            writer.writerow(header)
        print('Starting')

        while True:
            ret, frame = cam.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

            hand_landmarker_result = landmarker.detect_for_video(mp_image, int(cv2.getTickCount() / cv2.getTickFrequency() * 1000))

            if hand_landmarker_result.hand_landmarks:
                hand_landmarks = hand_landmarker_result.hand_landmarks[0]
                wrist = hand_landmarks[0]
                wrist_x, wrist_y, wrist_z = wrist.x, wrist.y, wrist.z

                normalized_coords = []
                for lm in hand_landmarks:
                    normalized_coords.extend([
                        lm.x - wrist_x,
                        lm.y - wrist_y,
                        lm.z - wrist_z
                    ])
                    cx, cy = int(lm.x * frame_Width), int(lm.y * frame_Height)
                    cv2.circle(frame, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
                frame_buffer.append(normalized_coords)

            cv2.imshow("Hand Landmarker", frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            if key in Gesture_Map and len(frame_buffer) == WINDOW_SIZE:
                label = Gesture_Map[key]
                flattened_window = []
                for frame in frame_buffer:
                    flattened_window.extend(frame)
                writer.writerow([label] + flattened_window)
                print(f"{n}Logged gesture {label}")
                n += 1


cam.release()
cv2.destroyAllWindows()