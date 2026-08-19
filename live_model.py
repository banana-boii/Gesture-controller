import cv2
import mediapipe as mp
from collections import deque
import joblib

WINDOW_SIZE = 10
task_model = "hand_landmarker.task"
prediction_model = "gesture_model.pkl"
clf = joblib.load(prediction_model)

cam = cv2.VideoCapture(0)
frame_Width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_Height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options = BaseOptions(model_asset_path = task_model),
    running_mode = VisionRunningMode.VIDEO,
    num_hands = 2
)

frame_buffer = deque(maxlen = WINDOW_SIZE)

with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cam.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        result = landmarker.detect_for_video(mp_image, int(cv2.getTickCount() / cv2.getTickFrequency() * 1000))

        current_gesture = "None"

        if result.hand_landmarks:
            hand_landmarks = result.hand_landmarks[0]
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

            if len(frame_buffer) == WINDOW_SIZE:
                flattened_window = []
                for frame_data in frame_buffer:
                    flattened_window.extend(frame_data)

                prediction = clf.predict([flattened_window])[0]
                current_gesture = prediction

        cv2.putText(frame, f"Gesture: {current_gesture}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow('Predicted gesture', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

cam.release()
cv2.destroyAllWindows()