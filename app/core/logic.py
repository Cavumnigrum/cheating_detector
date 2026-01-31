import time
import cv2
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
import os

@dataclass
class CheatingEvent:
    timestamp: float
    reason: str
    confidence: float
    evidence_path: str = ""

class CheatingDetector:
    def __init__(self):
        # --- CALIBRATION ---
        self.calibrated = False
        self.yaw_offset = 0.0
        self.pitch_offset = 0.0
        self.roll_offset = 0.0
        
        # --- STATE MACHINE ---
        self.state = "NORMAL" # NORMAL, SUSPICIOUS, ALERT
        self.suspicion_start_time = 0.0
        self.alert_start_time = 0.0
        self.last_keyboard_glance = 0.0
        
        # --- THRESHOLDS ---
        self.YAW_THRESH_SIDE = 35.0   
        self.PITCH_THRESH_DOWN = 25.0 
        self.TIME_TO_SUSPICIOUS = 1.0 
        self.TIME_TO_ALERT = 2.0      # Быстрое предупреждение, если сохраняется
        self.KEYBOARD_TIMEOUT = 2.0   
        
        # --- ВИДЕО БУФЕР ---
        self.video_buffer = deque(maxlen=150) # 5 секунд @ 30fps
        self.recording = False
        self.recording_frames = []
        self.post_alert_frame_count = 0
        self.POST_ALERT_FRAMES = 90 # 3 секунды
        
        # --- HISTORY ---
        self.events: List[CheatingEvent] = []

    def calibrate(self, yaw: float, pitch: float, roll: float):
        """Устанавливает 'нулевую' точку для положения головы."""
        self.yaw_offset = yaw
        self.pitch_offset = pitch
        self.roll_offset = roll
        self.calibrated = True
        print(f"[Logic] Calibrated: Yaw={yaw:.1f}, Pitch={pitch:.1f}")

    def process(self, frame: np.ndarray, phone_detected: bool, head_pose: Tuple[float, float, float], gaze_override: str = None, session_id: str = None) -> Dict:
        """
        Основной цикл логики
        """
        from app.core.logger import session_logger # Lazy import to avoid circular dependency
        
        current_time = time.time()
        pitch, yaw, roll = head_pose
        
        # 1. Нормализация углов
        rel_yaw = yaw - self.yaw_offset
        rel_pitch = pitch - self.pitch_offset
        rel_roll = roll - self.roll_offset
        
        # DEBUG: Показать углы
        print(f"Angle: Y {yaw:.0f} (Rel {rel_yaw:.0f}) | P {pitch:.0f} (Rel {rel_pitch:.0f}) | R {roll:.0f} (Rel {rel_roll:.0f}) | Calib:{self.calibrated}", flush=True)

        # Пороги
        YAW_THRESHOLD = 30  # Очень мягко
        PITCH_THRESHOLD = 20
        ROLL_THRESHOLD = 12
        
        # 2. Определение направления головы
        current_state = "Looking at Screen"
        
        if not self.calibrated:
             current_state = "Not Calibrated"
        else:
            if abs(rel_yaw) <= YAW_THRESHOLD and abs(rel_pitch) <= PITCH_THRESHOLD and abs(rel_roll) <= ROLL_THRESHOLD:
                current_state = "Looking at Screen"
            elif rel_yaw < -YAW_THRESHOLD:
                current_state = "Looking Right" 
            elif rel_yaw > YAW_THRESHOLD:
                current_state = "Looking Left"  
            elif rel_pitch > -PITCH_THRESHOLD:
                current_state = "Looking Down"  
            elif rel_pitch < PITCH_THRESHOLD:
                current_state = "Looking Up"    
            elif abs(rel_roll) > ROLL_THRESHOLD:
                current_state = "Tilted"
            
            if current_state != "Looking at Screen":
                 print(f"DEBUG: Violation! State={current_state} (Y:{rel_yaw:.0f} P:{rel_pitch:.0f})", flush=True)
                 
        # 2.5: Переопределение взглядом (Абсолютное - переопределяет проверки калибровки)
        if gaze_override:
            # Если глаза смотрят в сторону, переопределить "Looking at Screen" ИЛИ "Not Calibrated"
            # Но не переопределять явные движения Головы (если голова уже Смотрит Вправо, оставить так)
            if current_state in ["Looking at Screen", "Not Calibrated"]:
                current_state = gaze_override
                
            if current_state != "Looking at Screen":
                 print(f"DEBUG: Violation! State={current_state} (Y:{rel_yaw:.0f} P:{rel_pitch:.0f})", flush=True)
        
        # 3. Анализ поведения (Таймер смещения)
        reason = ""
        is_suspicious_now = False # Для логики записи
        
        # A. ТЕЛЕФОН
        # A. ТЕЛЕФОН
        if phone_detected:
            self.state = "CHEATING" 
            reason = "PHONE_CONFIRMED"
            is_suspicious_now = True
            print("DEBUG: [Logic] CHEATING CONFIRMED: Phone Detected", flush=True)
            
            # LOGGING
            if session_id:
                session_logger.log_event(session_id, "VIOLATION_PHONE", {"confidence": "high"})
            
        
        elif current_state != "Looking at Screen" and (self.calibrated or gaze_override):
            if self.suspicion_start_time == 0:
                self.suspicion_start_time = current_time
                self.state = "SUSPICIOUS"
                if session_id:
                        session_logger.log_event(session_id, "VIOLATION_GAZE_SUSPICIOUS", {"state": current_state})

            elif current_time - self.suspicion_start_time >= 3.0:
                if self.state != "ALERT":
                     self.state = "ALERT"
                     if session_id:
                         session_logger.log_event(session_id, "VIOLATION_GAZE_ALERT", {"state": current_state, "duration": 3.0})

                reason = f"PROLONGED_{current_state.upper().replace(' ', '_')}"
                is_suspicious_now = True
        else:
            # Обратно в норму
            # Если в данный момент не заблокировано в состояниях высокой тревоги
            if self.state not in ["ALERT", "CHEATING"]: 
                self.state = "NORMAL"
                self.suspicion_start_time = 0
            else:
                 # Если в ALERT/CHEATING, логика в 'Record Logic' обработает остывание/выход
                 self.suspicion_start_time = 0

        # 4. Обработка переходов состояний и запись
        status = {
            "state": self.state,
            "gaze_zone": current_state,
            "reason": reason,
            "calibrated": self.calibrated
        }

        # Логика видео буфера
        # Всегда добавлять в пре-буфер
        self.video_buffer.append(frame.copy())
        
        # Запуск записи при ALERT или CHEATING
        if self.state in ["ALERT", "CHEATING"]:
            if not self.recording:
                # НАЧАЛО ЗАПИСИ
                self.recording = True
                self.recording_frames = list(self.video_buffer) # Сброс пре-буфера
                print(f"[Logic] Evidence Recording Started: {reason}")
            
            # Продолжение записи
            self.recording_frames.append(frame.copy())
            
            # Проверка, нужно ли остановиться (если угроза миновала)
            if not is_suspicious_now:
                # Мы в ALERT, но прямая угроза ушла. 
                # Нам нужно остывание (3 секунды)
                self.post_alert_frame_count += 1
                if self.post_alert_frame_count > self.POST_ALERT_FRAMES:
                    # ОСТАНОВКА ЗАПИСИ
                    self.save_evidence()
                    self.state = "NORMAL"
                    self.recording = False
                    self.post_alert_frame_count = 0
            else:
                self.post_alert_frame_count = 0 # Сброс остывания, если угроза появляется снова
                
        # Отладка финального статуса
        if self.state != "NORMAL":
             print(f"DEBUG: Logic Return: {status['state']} (Reason: {status['reason']})", flush=True)
                
        return status

    def save_evidence(self):
        # Сохранение self.recording_frames на диск
        timestamp = int(time.time())
        filename = f"evidence_{timestamp}.avi"
        # Убедиться, что папка логов существует
        log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "evidence")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
            
        path = os.path.join(log_dir, filename)
        
        if not self.recording_frames: return

        height, width, layers = self.recording_frames[0].shape
        out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'MJPG'), 30, (width, height))
        
        for f in self.recording_frames:
            out.write(f)
        out.release()
        
        self.events.append(CheatingEvent(time.time(), "ALERT_RECORDED", 1.0, path))
        print(f"[Logic] Evidence saved to {path}")
        self.recording_frames = []
