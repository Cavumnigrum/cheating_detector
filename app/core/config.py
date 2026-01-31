from pydantic import BaseConfig

class Settings(BaseConfig):
    PROJECT_NAME: str = "Phone Detection AI"
    API_V1_STR: str = "/api/v1"
    # Основная (Ultimate) модель (Roboflow + COCO Phone)
    MODEL_PATH: str = "runs/detect/yolo11_ultimate_v3/weights/best.pt"
    # MODEL_PATH: str = "yolo11n.pt" # Резервный вариант для тестирования
    
    # Флаги функций
    USE_SCENE_CLASSIFIER: bool = False 

settings = Settings()
