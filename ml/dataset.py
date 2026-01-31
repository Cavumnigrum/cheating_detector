from ultralytics.utils.downloads import download
from pathlib import Path
import shutil
import yaml
import os

def prepare_phone_dataset(base_path: str = "d:/labs/phone_detecter/dataset"):
    """
    Загружает COCO128, фильтрует класс 'мобильный телефон' (ID 67),
    и создает новую структуру датасета, где класс 0 = телефон.
    """
    base = Path(base_path)
    if base.exists():
        print(f"Dataset folder {base} already exists. Skipping download/prep.")
        return str(base / "data.yaml")

    print("Downloading COCO128...")
    # Загрузка coco128.zip в текущую папку (извлекается в ./coco128)
    # url = "https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip"
    
    # Скачаем вручную, чтобы быть уверенными
    zip_path = Path("coco128.zip")
    if not zip_path.exists() and not Path("coco128").exists():
        download("https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip")
    
    # Обычно извлекается в 'datasets/coco128' или './coco128' в зависимости от конфига.
    # Предположим ./coco128 для простоты, если мы распакуем его. 
    # Функция загрузки из ultralytics может поместить его в определенное место.
    # Чтобы быть уверенными, поищем его.
    
    source_dir = Path("datasets/coco128")
    if not source_dir.exists():
        source_dir = Path("coco128")
    
    if not source_dir.exists():
        # Резервный вариант: попытаться распаковать, если zip существует
        import zipfile
        if zip_path.exists():
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(".")
            source_dir = Path("coco128")
    
    if not source_dir.exists():
        raise FileNotFoundError("Could not find coco128 dataset after download.")

    print(f"Filtering dataset from {source_dir} to {base}...")
    
    # Структура
    (base / "images" / "train").mkdir(parents=True, exist_ok=True)
    (base / "labels" / "train").mkdir(parents=True, exist_ok=True)
    
    # Класс COCO 67 - это мобильный телефон. Мы отображаем его в 0.
    COCO_PHONE_CLASS = 67
    
    # Обработка изображений и меток
    # COCO128 содержит images/train2017 и labels/train2017
    src_images = source_dir / "images" / "train2017"
    src_labels = source_dir / "labels" / "train2017"
    
    if not src_images.exists():
        # Возможно, плоская структура?
        src_images = source_dir / "images"
        src_labels = source_dir / "labels"

    count = 0
    for label_file in src_labels.glob("*.txt"):
        with open(label_file, "r") as f:
            lines = f.readlines()
        
        new_lines = []
        has_phone = False
        for line in lines:
            parts = line.strip().split()
            if len(parts) > 0:
                cls_id = int(parts[0])
                if cls_id == COCO_PHONE_CLASS:
                    # Переназначение в 0
                    parts[0] = "0"
                    new_lines.append(" ".join(parts))
                    has_phone = True
        
        if has_phone:
            # Копирование изображения
            img_name = label_file.stem + ".jpg" # Предполгаем jpg
            src_img = src_images / img_name
            if src_img.exists():
                shutil.copy(src_img, base / "images" / "train" / img_name)
                # Запись новой метки
                with open(base / "labels" / "train" / label_file.name, "w") as f:
                    f.write("\n".join(new_lines))
                count += 1
    
    print(f"Extracted {count} images containing phones.")
    
    # Создание data.yaml
    yaml_content = {
        "path": base.absolute().as_posix(),
        "train": "images/train",
        "val": "images/train", # Использовать то же самое для валидации в этом маленьком примере
        "names": {0: "phone"}
    }
    
    yaml_path = base / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f)
        
    return str(yaml_path)

if __name__ == "__main__":
    prepare_phone_dataset()
