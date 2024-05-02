import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
import time
import os

# For static images:
path = "C:\\Users\\RBC\\Documents\\School\\Cornell\\EmPRISE\\patient_on_bed\\taxonomy"
IMAGE_FILES = []
IMAGE_NAMES = []
for filename in os.listdir(path):
    # Check if the file is an image file (you can add more image extensions if needed)
    if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        # Construct the full path to the image file
        image_path = os.path.join(path, filename)
        IMAGE_NAMES.append(filename)
        IMAGE_FILES.append(image_path)
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
    # Convert the BGR image to RGB before processing.
    results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if results.pose_landmarks:
      with open("landmarks_taxonomy.txt", "a") as f:
      
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
    # Draw segmentation on the image.
    # To improve segmentation around boundaries, consider applying a joint
    # bilateral filter to "results.segmentation_mask" with "image".
    # if(results.segmentation_mask is not None):
    #   condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
    #   bg_image = np.zeros(image.shape, dtype=np.uint8)
    #   bg_image[:] = BG_COLOR
    #   annotated_image = np.where(condition, annotated_image, bg_image)
      # Draw pose, left and right hands, and face landmarks on the image.
    mp_drawing.draw_landmarks(
        annotated_image,
        results.face_landmarks,
        mp_holistic.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles
        .get_default_face_mesh_tesselation_style())
    mp_drawing.draw_landmarks(
        annotated_image,
        results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.
        get_default_pose_landmarks_style())
    # cv2.imshow('annotate', annotated_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    annotated_path = "C:\\Users\\RBC\\Documents\\School\\Cornell\\EmPRISE\\patient_on_bed\\annotate_taxonomy"
    img_path = os.path.join(annotated_path, IMAGE_NAMES[idx])
    print(img_path)
    cv2.imwrite(img_path , annotated_image)
    # Plot pose world landmarks.
    # mp_drawing.plot_landmarks(
    #     results.pose_world_landmarks, mp_holistic.POSE_CONNECTIONS)
    with open("pose_world_landmarks_tax.txt", "a") as f:
        f.write(
        f'{idx}'
        f'Pose Landmarks 3d: (\n'
        f'{results.pose_world_landmarks}\n\n')