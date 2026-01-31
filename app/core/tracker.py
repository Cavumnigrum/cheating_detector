
from .logic import CheatingDetector
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import os

class BehaviorTracker:
    def __init__(self):
        # Настройка API задач MediaPipe
        model_path = os.path.join(os.path.dirname(__file__), 'face_landmarker.task')
        
        # Проверка существования модели
        if not os.path.exists(model_path):
            print(f"WARNING: Face Landmarker model not found at {model_path}. Please download it.")
        
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        
        self.logic = CheatingDetector()
        self.alerts_history = [] 
        self.calibration_requested = False
        
        # История сглаживания
        from collections import deque
        self.yaw_history = deque(maxlen=10)
        self.pitch_history = deque(maxlen=10)
        self.roll_history = deque(maxlen=10)
        self.iris_history = deque(maxlen=10) 

    def trigger_calibration(self):
        self.calibration_requested = True
 

    def process_frame(self, frame_bgr, phone_detected=False, session_id=None):
        h, w, _ = frame_bgr.shape
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        # Create MP Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Detect
        detection_result = self.landmarker.detect(mp_image)
        
        head_pose = (0, 0, 0)
        landmarks_detected = False
        gaze_override = None
        if detection_result.face_landmarks:
            landmarks_detected = True
            landmarks = detection_result.face_landmarks[0] # List of NormalizedLandmark
            
            # --- ПОЛОЖЕНИЕ ГОЛОВЫ ---
            face_2d = []
            
            # Индексы MediaPipe: [Нос, Подбородок, Левый Глаз, Правый Глаз, Левый Рот, Правый Рот]
            points_idx = [1, 152, 33, 263, 61, 291]
            
            # Точки 2D изображения (Обнаруженные)
            for idx in points_idx:
                lm = landmarks[idx]
                x, y = int(lm.x * w), int(lm.y * h)
                face_2d.append([x, y])
            
            face_2d = np.array(face_2d, dtype=np.float64)

            # Точки 3D модели (Обобщенное человеческое лицо)
            # Используется как эталон для вычисления вращения
            # X: Влево/Вправо (Отрицательный Влево)
            # Y: Вверх/Вниз (Отрицательный Вверх, Положительный Вниз) -> конвенция OpenCV
            # Z: Вперед/Назад (Отрицательный Вперед)
            face_3d = np.array([
                (0.0, 0.0, 0.0),             # Кончик носа
                (0.0, 330.0, -65.0),         # Подбородок (Вниз = +Y)
                (-225.0, -170.0, -135.0),    # Левый глаз левый угол (Вверх = -Y)
                (225.0, -170.0, -135.0),     # Правый глаз правый угол (Вверх = -Y)
                (-150.0, 150.0, -125.0),     # Левый угол рта (Вниз = +Y)
                (150.0, 150.0, -125.0)       # Правый угол рта (Вниз = +Y)
            ], dtype=np.float64)

            focal_length = 1 * w
            cam_matrix = np.array([[focal_length, 0, w / 2],
                                   [0, focal_length, h / 2],
                                   [0, 0, 1]])
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
            gaze_override = None

            if success:
                rmat, _ = cv2.Rodrigues(rot_vec)
                sy = np.sqrt(rmat[0, 0] * rmat[0, 0] + rmat[1, 0] * rmat[1, 0])
                if sy < 1e-6:
                     pitch = np.arctan2(-rmat[1, 2], rmat[1, 1])
                     yaw = np.arctan2(-rmat[2, 0], sy)
                     roll = 0
                else:
                     pitch = np.arctan2(rmat[2, 1], rmat[2, 2])
                     yaw = np.arctan2(-rmat[2, 0], sy)
                     roll = np.arctan2(rmat[1, 0], rmat[0, 0])
                
                # Сырые углы из PnP
                p = np.degrees(pitch)
                y = np.degrees(yaw)
                # r = np.degrees(roll) # PnP Крен может быть нестабильным
                
                # Геометрический крен (Надежный)
                # Левый Глаз (33) -> Правый Глаз (263)
                # В списке ориентиров: 33 это индекс 2, 263 это индекс 3 в нашем 'points_idx'
                # points_idx = [1, 152, 33, 263, 61, 291]
                #              0    1   2    3
                
                # На самом деле, используем сырые ориентиры для точности
                lm_left_eye = landmarks[33]
                lm_right_eye = landmarks[263]
                
                dY = (lm_right_eye.y - lm_left_eye.y)
                dX = (lm_right_eye.x - lm_left_eye.x)
                # Примечание: Y направлен вниз на изображении, поэтому если Правый Глаз "ниже" (больше Y), dY > 0.
                # Если dY > 0, голова наклонена вправо (по часовой стрелке). 
                # atan2(y, x) -> положительно для вращения по ЧС, если Y направлен вниз.
                geo_roll = np.degrees(np.arctan2(dY, dX))
                
                # Сглаживание углов
                self.pitch_history.append(p)
                self.yaw_history.append(y)
                self.roll_history.append(geo_roll)
                
                smooth_p = np.mean(self.pitch_history)
                smooth_y = np.mean(self.yaw_history)
                smooth_r = np.mean(self.roll_history)
                
                head_pose = (smooth_p, smooth_y, smooth_r)
                
                # --- ОТСЛЕЖИВАНИЕ ВЗГЛЯДА (ЗРАЧОК) ---
                # Ориентиры: 468 (Левый Зрачок), 473 (Правый Зрачок)
                # Углы Левого Глаза: 33 (Внутренний), 133 (Внешний) - Координаты Изображения (Лево=33, Право=133)
                # Углы Правого Глаза: 362 (Внутренний), 263 (Внешний) - Координаты Изображения (Лево=362, Право=263)
                
                gaze_override = None
                
                # Проверка наличия ориентиров зрачка
                # print(f"DEBUG: Landmarks Len: {len(landmarks)}", flush=True)
                
                if len(landmarks) > 468:
                    try:
                        # Нормализация координат x
                        
                        # Левый глаз (На изображении слева)
                        l_left = landmarks[33].x   # Внутренний/Левый край
                        l_right = landmarks[133].x # Внешний/Правый край
                        l_center = landmarks[468].x
                        l_width = l_right - l_left
                        if l_width > 0:
                            l_ratio = (l_center - l_left) / l_width
                        else:
                            l_ratio = 0.5

                        # Правый глаз (На изображении справа)
                        r_left = landmarks[362].x  # Внутренний/Левый край
                        r_right = landmarks[263].x # Внешний/Правый край
                        r_center = landmarks[473].x
                        r_width = r_right - r_left
                        if r_width > 0:
                            r_ratio = (r_center - r_left) / r_width
                        else:
                            r_ratio = 0.5
                        
                        raw_avg_ratio = (l_ratio + r_ratio) / 2.0
                        self.iris_history.append(raw_avg_ratio)
                        # сглаживание по последним 5 кадрам
                        avg_ratio = np.mean(list(self.iris_history)[-5:])
                        
                        if avg_ratio < 0.375: 
                             gaze_override = "Looking Right" # Справа на изображении
                        elif avg_ratio > 0.625:
                             gaze_override = "Looking Left"  # Слева на изображении
                             
                        # DEBUG: Включите это для настройки порогов
                        print(f"DEBUG: Eye Ratio: {avg_ratio:.2f} (L:{l_ratio:.2f} R:{r_ratio:.2f}) override={gaze_override}", flush=True)
                        
                    except Exception as e:
                        print(f"DEBUG: Iris Logic Error: {e}", flush=True)

        # --- ПРОВЕРКА КАЛИБРОВКИ ---
        if self.calibration_requested and landmarks_detected:
            self.logic.calibrate(*head_pose)
            self.calibration_requested = False

        # --- ЛОГИКА ОБНОВЛЕНИЯ ---
        status = self.logic.process(frame_bgr, phone_detected, head_pose, gaze_override, session_id)
        
        if status['reason']:
             self._add_alert(status['reason'], status['state'])
        
        ui_score = 10
        if status['state'] == 'SUSPICIOUS': ui_score = 60
        elif status['state'] == 'ALERT': ui_score = 95
        elif status['state'] == 'CHEATING': ui_score = 100
        
        # Конвертация ориентиров в список для фронтенда
        landmarks_list = []
        if landmarks_detected:
            # У нас уже есть 'landmarks' (список NormalizedLandmark)
            # Конвертировать в [{x, y, z}]
            for lm in landmarks:
                landmarks_list.append({"x": lm.x, "y": lm.y, "z": lm.z})

        return {
            "head_pose": head_pose,
            "state": status['state'],
            "message": status['reason'] or "Monitoring...",
            "score": ui_score,
            "history": self.alerts_history[-5:],
            "landmarks_detected": landmarks_detected,
            "landmarks": landmarks_list # Новое поле
        }

    def _add_alert(self, reason, state):
        timestamp = time.time()
        if self.alerts_history and (timestamp - self.alerts_history[-1]['timestamp'] < 2.0) and self.alerts_history[-1]['code'] == reason:
            return

        severity = 50
        if state == 'ALERT': severity = 90
        
        self.alerts_history.append({
            "code": reason,
            "message": reason,
            "severity": severity,
            "timestamp": timestamp
        })
