from ultralytics import YOLO
from roboflow import Roboflow
import os
import shutil
import glob
from pathlib import Path

# --- КОНФИГУРАЦИЯ ---
# ROBOFLOW_API_KEY теперь извлекается из os.getenv("ROBOFLOW_API_KEY")
PROJECT_ROOT = Path("d:/labs/phone_detecter")
DATASETS_DIR = PROJECT_ROOT / "datasets"
COCO_PHONE_DIR = DATASETS_DIR / "coco128_phone"

def main():
    print("--- STARTING ULTIMATE DATASET PREPARATION (V3) ---")
    
    # ---------------------------------------------------------
    # 1. НАСТРОЙКА ИСТОЧНИКОВ
    # ---------------------------------------------------------
    
    # A. ROBOLFLOW
    print(">> [1/4] Подготовка данных Roboflow...")
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        print("ERROR: ROBOFLOW_API_KEY environment variable not set.")
        return

    rf = Roboflow(api_key=api_key)
    project = rf.workspace("d1156414").project("cellphone-0aodn")
    dataset = project.version(project.versions()[-1].version).download("yolov11", location=str(PROJECT_ROOT / "roboflow_dataset"))
    roboflow_root = Path(dataset.location)
    
    # B. COCO128
    print(">> [2/4] Подготовка данных COCO128...")
    coco_path = find_coco128()
    if coco_path:
        prepare_coco_phone(coco_path, COCO_PHONE_DIR)
    else:
        print("!! COCO128 не найден. Пропуск.")

    # C. FPI ДАТАСЕТ
    print(">> [3/4] Подготовка FPI датасета...")
    fpi_root = PROJECT_ROOT / "datasets/reorganized_dataset"
    if not fpi_root.exists():
        print(f"!! FPI датасет не найден в {fpi_root}. Пропуск.")
    else:
        print(f"Найден FPI в {fpi_root}")

    # ---------------------------------------------------------
    # 2. АГРЕГАЦИЯ ПУТЕЙ
    # ---------------------------------------------------------
    print(">> [4/4] Сборка конфигурации...")
    
    # Инициализация списков (КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ)
    train_paths = []
    val_paths = []
    
    # Добавление Roboflow (Всегда существует, если шаг A успешен)
    train_paths.append(str(roboflow_root / "train/images"))
    val_paths.append(str(roboflow_root / "valid/images"))
    
    # Добавление COCO
    if coco_path and COCO_PHONE_DIR.exists():
        train_paths.append(str(COCO_PHONE_DIR / "images/train2017"))
        
    # Добавление FPI
    if fpi_root.exists():
        train_paths.append(str(fpi_root / "train/images"))
        val_paths.append(str(fpi_root / "val/images"))
        # Тестовый набор FPI? Мы можем добавить в вал или оставить для тестирования.
        # Добавим тест в вал для максимальной информации о валидации во время обучения
        val_paths.append(str(fpi_root / "test/images"))

    # ---------------------------------------------------------
    # 3. СОЗДАНИЕ YAML
    # ---------------------------------------------------------
    ultimate_yaml_path = PROJECT_ROOT / "ml/ultimate.yaml"
    
    yaml_content = f"path: {str(PROJECT_ROOT).replace(os.sep, '/')}\n"
    yaml_content += "train:\n"
    for p in train_paths:
        yaml_content += f"  - {p.replace(os.sep, '/')}\n"
        
    yaml_content += "val:\n"
    for p in val_paths:
        yaml_content += f"  - {p.replace(os.sep, '/')}\n"
        
    # Определение классов. FPI имеет "Face" как класс 1 (вероятно). Roboflow только Phone (класс 0).
    # Определяем оба, чтобы избежать ошибок, но нас интересует 0.
    yaml_content += "names:\n"
    yaml_content += "  0: phone\n"
    yaml_content += "  1: face_secondary\n"
    
    with open(ultimate_yaml_path, "w") as f:
        f.write(yaml_content)
        
    print(f"YAML создан: {ultimate_yaml_path}")
    print(f"Обучение на: {len(train_paths)} источниках изображений.")

    # ---------------------------------------------------------
    # 4. ОБУЧЕНИЕ
    # ---------------------------------------------------------
    print(">> НАЧАЛО ОБУЧЕНИЯ (YOLOv11n)...")
    
    model = YOLO("yolo11n.pt")
    
    # Обучение
    results = model.train(
        data=str(ultimate_yaml_path),
        epochs=30,
        imgsz=640,
        batch=8, # Лимит безопасности для 6GB VRAM
        device=0,
        name="yolo11_ultimate_v3",
        exist_ok=True 
    )
    
    print("--- ОБУЧЕНИЕ ЗАВЕРШЕНО ---")
    if hasattr(results, 'best'): 
        print(f"Лучшая модель: {results.best}")
    else:
        # Резервный вариант, если атрибут отсутствует (старый ultralytics?)
        print(f"Лучшая модель должна быть в: runs/detect/yolo11_ultimate_v3/weights/best.pt")


def find_coco128():
    # Помощник для поиска COCO128
    try:
        YOLO("yolo11n.pt").check_dataset("coco128.yaml") 
    except: pass
    
    potential_paths = [
        PROJECT_ROOT / "coco128", 
        Path("coco128"),
        Path("datasets/coco128"),
        PROJECT_ROOT / "datasets/coco128"
    ]
    for p in potential_paths:
        if p.exists(): return p
    return None

def prepare_coco_phone(coco_path, output_dir):
    # Извлекает класс 67 (Телефон) из COCO128 в класс 0
    if output_dir.exists():
        shutil.rmtree(output_dir)
    
    (output_dir / "images/train2017").mkdir(parents=True)
    (output_dir / "labels/train2017").mkdir(parents=True)
    
    src_labels = list((coco_path / "labels/train2017").glob("*.txt"))
    count = 0
    
    for lbl_file in src_labels:
        with open(lbl_file, "r") as f:
            lines = f.readlines()
            
        new_lines = []
        has_phone = False
        for line in lines:
            parts = line.strip().split()
            if not parts: continue
            cls_id = int(parts[0])
            if cls_id == 67: # Мобильный телефон
                has_phone = True
                new_lines.append(f"0 {' '.join(parts[1:])}\n")
                
        if has_phone:
            with open(output_dir / "labels/train2017" / lbl_file.name, "w") as f:
                f.writelines(new_lines)
            
            # Поиск и копирование изображения
            img_name = lbl_file.stem
            for ext in [".jpg", ".jpeg", ".png"]:
                img_src = coco_path / "images/train2017" / (img_name + ext)
                if img_src.exists():
                    shutil.copy(img_src, output_dir / "images/train2017" / (img_name + ext))
                    break
            count += 1
            
    print(f"Извлечено {count} изображений телефонов из COCO128.")

if __name__ == "__main__":
    main()
