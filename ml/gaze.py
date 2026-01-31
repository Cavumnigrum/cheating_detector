import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class GazeDetector:
    def __init__(self):
        # Создание объекта FaceLandmarker.
        model_path = r"d:\labs\phone_detecter\face_landmarker.task"
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1)
        self.detector = vision.FaceLandmarker.create_from_options(options)
        
    def detect_gaze(self, image_bytes: bytes):
        """
        Анализ изображения на направление взгляда с использованием FaceLandmarker.
        """
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            h, w, _ = img.shape
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Конвертация в MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            
            # Обнаружение
            detection_result = self.detector.detect(mp_image)
            
            if not detection_result.face_landmarks:
                return {"status": "NO_FACE", "score": 1.0}
            
            # ... (previous code)
            
            # Blendshapes (Микровыражения)
            categories = detection_result.face_blendshapes[0]
            # Отображение имени категории в оценку
            blendshapes = {cat.category_name: cat.score for cat in categories}
            
            # Взгляд глаз (Влево/Вправо)
            # "In" означает взгляд к носу, "Out" - взгляд в сторону
            # Левый глаз: In=Вправо, Out=Влево (с точки зрения пользователя)
            # Правый глаз: In=Влево, Out=Вправо
            
            # Простые агрегации
            eye_look_left = blendshapes.get('eyeLookOutLeft', 0) + blendshapes.get('eyeLookInRight', 0)
            eye_look_right = blendshapes.get('eyeLookInLeft', 0) + blendshapes.get('eyeLookOutRight', 0)
            
            # Положение головы (На основе геометрии - сохранено для надежности)
            # ... (повторное использование логики геометрии) ...
            landmarks = detection_result.face_landmarks[0]
            nose = landmarks[1].x
            left_eye_outer = landmarks[33].x
            right_eye_outer = landmarks[263].x
            
            # Нормализованная позиция носа (от -1 до 1, где 0 - центр)
            face_width = abs(left_eye_outer - right_eye_outer)
            if face_width == 0: head_pos = 0
            else:
                center = (left_eye_outer + right_eye_outer) / 2
                head_pos = (nose - center) / face_width / 0.5 # Масштабирование до ~ -1..1
            
            return {
                "status": "DETECTED",
                "head_pos": head_pos, # < -0.3 Вправо, > 0.3 Влево
                "eye_left": eye_look_left,
                "eye_right": eye_look_right,
                "raw_blendshapes": blendshapes
            }
        except Exception as e:
            print(f"Gaze Error: {e}")
            return {"status": "ERROR"}
