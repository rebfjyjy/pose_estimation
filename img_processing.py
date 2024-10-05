import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
import time
import os

# For static images:
path = "./6.11"
IMAGE_FILES = ['./6.11/rgb_image.png']
IMAGE_NAMES = []
# for filename in os.listdir(path):
#     # Check if the file is an image file (you can add more image extensions if needed)
#     if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
#         # Construct the full path to the image file
#         image_path = os.path.join(path, filename)
#         IMAGE_NAMES.append(filename)
#         IMAGE_FILES.append(image_path)
BG_COLOR = (192, 192, 192) # gray
with mp_holistic.Holistic(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=False,
    refine_face_landmarks=True) as holistic:
  for idx, file in enumerate(IMAGE_FILES):
    image = cv2.imread(file)
    # cv2.imshow('image', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    image_height, image_width, _ = image.shape
    print(image_height, image_width)
    # Convert the BGR image to RGB before processing.
    results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    print(results.pose_landmarks)
    if results.pose_landmarks:
      with open("./landmarks.txt", "a") as f:
      
        f.write(
        f'Nose coordinates: ('
        f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].x * image_width}, '
        f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].y * image_height})\n'
        f'LEFT_EYE: ('
        f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EYE].x * image_width}, '
        f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EYE].y * image_height})\n'
        f'RIGHT_EYE: ('
        f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EYE].x * image_width}, '
        f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EYE].y * image_height})\n'
        f'LEFT_EAR: ('
        f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x * image_width}, '
        f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y * image_height})\n'
        f'RIGHT_EAR: ('
        f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EAR].x * image_width}, '
        f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EAR].y * image_height})\n'
        f'MOUTH_LEFT: ('
        f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.MOUTH_LEFT].x * image_width}, '
        f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.MOUTH_LEFT].y * image_height})\n'
        f'MOUTH_RIGHT: ('
        f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.MOUTH_RIGHT].x * image_width}, '
        f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.MOUTH_RIGHT].y * image_height})\n'
        f'LEFT_SHOULDER: ('
        f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].x * image_width}, '
        f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].y * image_height})\n\n')
        

    annotated_image = image.copy()
    mp_drawing.draw_landmarks(
        annotated_image,
        results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.
        get_default_pose_landmarks_style())
    cv2.imshow('annotate', annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    annotated_path = "./6.11"
    img_path = './6.11/rgb_image'
    cv2.imwrite(f'{img_path}_annotated.png' , annotated_image)
    holistic.close()
