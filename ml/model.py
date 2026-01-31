from ultralytics import YOLO
import cv2
import numpy as np

class PhoneDetector:
    def __init__(self, model_path: str = "yolo11n.pt"):
        """
        Инициализация детектора YOLOv11.
        Используется базовая модель, которая включает 'мобильный телефон' (класс 67).
        """
        self.model = YOLO(model_path) # Загрузка стандартной модели

    def predict_image_object(self, image_bytes: bytes, conf: float = 0.4):
        """Запуск инференса на байтах изображения в памяти."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        return self._process_results(img, conf)

    def _process_results(self, img, conf):
        """
        Предсказание на классах COCO, конкретно класс 67 (мобильный телефон).
        """
        # DEBUG: Печать размера ввода
        print(f"DEBUG: Processing image {img.shape}", flush=True)
        
        # DEBUG MODE: Обнаружение ВСЕХ классов, чтобы видеть происходящее
        results = self.model.predict(img, conf=conf, verbose=False) 
        
        if not results:
            print("DEBUG: No results object returned", flush=True)
            
        detections = []
        for result in results:
            print(f"DEBUG: Found {len(result.boxes)} boxes", flush=True)
            for box in result.boxes:
                conf_val = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = result.names[cls_id]
                

                if cls_id == 67 or label.lower() in ["cell phone", "phone", "mobile phone", "smartphone", "phone"] and conf_val>=0.75:
                    print(f"DEBUG: Phone Detected! ({conf_val:.2f})", flush=True)
                    detections.append({
                        "bbox": box.xyxy[0].tolist(),
                        "conf": conf_val,
                        "cls": cls_id,
                        "label": "Phone (Cheating)"
                    })
                else:

                    print(f"DEBUG: Ignored {label} ({conf_val:.2f})", flush=True)
                    pass

        return detections
