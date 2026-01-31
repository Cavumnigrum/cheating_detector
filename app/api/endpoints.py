from fastapi import APIRouter, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from app.core.config import settings
from ml.model import PhoneDetector
from ml.gaze import GazeDetector
from app.core.tracker import BehaviorTracker
from app.core.logger import session_logger
import uuid
import json

router = APIRouter()

# Инициализация моделей (Глобальные, так как они тяжелые и stateless)
detector = PhoneDetector(settings.MODEL_PATH)
gaze_detector = GazeDetector() 

@router.post("/detect")
async def detect_phones(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    contents = await file.read()
    detections = detector.predict_image_object(contents)
    return {"filename": file.filename, "detections": detections}

@router.websocket("/ws/detect")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # Session Setup
    session_id = str(uuid.uuid4())
    client_ip = websocket.client.host if websocket.client else "unknown"
    
    # Per-Session Tracker (Isolates state per user)
    local_tracker = BehaviorTracker()
    
    # Log Start
    session_logger.log_session_start(session_id, client_ip)
    print(f"Session Started: {session_id} ({client_ip})")
    
    try:
        while True:
            # Обработка текста (команды) или байтов (изображения)
            message = await websocket.receive()
            
            if "text" in message:
                 # Обработка команд (например, {"type": "calibrate"})
                 msg_data = json.loads(message["text"])
                 if msg_data.get("type") == "calibrate":
                     print(f"Received Calibration Request [{session_id}]")
                     local_tracker.trigger_calibration()
                 continue
            
            if "bytes" not in message:
                continue
                
            data = message["bytes"]
            
            # Декодирование изображения в BGR (OpenCV)
            import numpy as np
            import cv2
            nparr = np.frombuffer(data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None: 
                print("Error: Decoded img is None", flush=True)
                continue

            # Heartbeat (Подтверждение активности)
            if not hasattr(websocket, 'frame_count'): websocket.frame_count = 0
            websocket.frame_count += 1
            if websocket.frame_count % 30 == 0:
                print(f"DEBUG: Processed {websocket.frame_count} frames", flush=True)
            
            # 1. Обнаружение телефона
            phone_results = detector._process_results(img, conf=0.3) # Снижено до 0.3
            phone_detected = len(phone_results) > 0
            
            # 2. Анализ поведения (теперь включает Face Mesh)
            # Передаем session_id для логирования событий
            behavior_status = local_tracker.process_frame(img, phone_detected, session_id=session_id)
            
            # DEBUG: Печать статуса
            if phone_detected: print(f"Phone Detected! {len(phone_results)}", flush=True)

            # Объединение
            response = {
                "detections": phone_results,
                "behavior": behavior_status
            }
            
            await websocket.send_json(response)
            
    except WebSocketDisconnect:
        print(f"Client disconnected: {session_id}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"WS Error: {e}")
        try:
            await websocket.close()
        except:
            pass
    finally:
        # Log End
        session_logger.log_session_end(session_id)
        print(f"Session Ended: {session_id}")
