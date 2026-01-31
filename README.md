# Phone & Gaze Detection AI / ИИ для обнаружения телефона и взгляда

[English Version](#english-version) | [Русская Версия](#русская-версия)

---

## English Version

### Project Overview

This project is an advanced **Proctoring AI System** designed to detect cheating behaviors in real-time. It uses a combination of Computer Vision models to monitor user activity via a webcam.

### Key Features

1.  **Phone Detection**: Uses a custom-trained **YOLOv11** model to detect mobile phones with high precision, specifically aiming to reduce false negatives on hand-held phones.
2.  **Gaze Tracking (Iris & Head Pose)**:
    - **Head Pose**: Tracks Pitch, Yaw, and Roll to detect if the user is looking away from the screen.
    - **Iris Tracking**: Uses **MediaPipe Face Mesh** (478 landmarks) to track pupil position relative to eye corners. It can detect if a user is looking sideways _without_ turning their head.
3.  **Behavior Analysis Logic**:
    - **State Machine**: Tracks states (`NORMAL`, `SUSPICIOUS`, `ALERT`, `CHEATING`) with time-based hysteresis (e.g., looking away for >3 seconds triggers an alert).
    - **Evidence Recording**: Automatically records short video clips (Server-Side) when cheating is suspected or confirmed.
4.  **Real-Time WebSocket API**: Low-latency communication between the frontend and the Python backend.

### Datasets Used

- **Roboflow Universe**: [Cellphone Computer Vision Model](https://universe.roboflow.com/d1156414/cellphone-0aodn) - "Phone on hand" dataset for detection in difficult angles.
- **COCO128**: Standard dataset filtered for class `67` (Cell Phone) to maintain baseline performance.
- **FPI (Face Phone Interaction)**: [Dataset from arXiv:2509.09111v1](https://arxiv.org/html/2509.09111v1) - Used for analysis of specific face-phone interaction behaviors.

### Installation

1.  **Clone the repository**:

    ```bash
    git clone <repo_url>
    cd phone_detecter
    ```

2.  **Set up Virtual Environment**:

    ```bash
    python -m venv venv
    ./venv/Scripts/Activate.ps1
    ```

3.  **Install Dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Environment Variables** (for training):
    - Create a `.env` file or set in system: `ROBOFLOW_API_KEY=your_key`

### Usage

1.  **Run the Server**:

    ```powershell
    ./run_server.bat
    ```

    - Or manually: `uvicorn app.api.endpoints:app --reload`

2.  **Open the Web App**:
    - Navigate to `http://localhost:8000`.

---

## Русская Версия

### Обзор Проекта

Этот проект представляет собой продвинутую **ИИ-систему прокторинга**, разработанную для обнаружения подозрительного поведения в реальном времени. Система использует комбинацию моделей компьютерного зрения для мониторинга пользователя через веб-камеру.

### Ключевые Возможности

1.  **Обнаружение Телефона**: Использует специально обученную модель **YOLOv11** для высокоточного обнаружения мобильных телефонов, с особым акцентом на телефоны в руках (сложные ракурсы).
2.  **Трекинг Взгляда (Зрачок и Голова)**:
    - **Положение Головы**: Отслеживает наклон и поворот (Pitch, Yaw, Roll) для фиксации отворота от экрана.
    - **Трекинг Зрачка**: Использует **MediaPipe Face Mesh** (478 точек) для отслеживания положения зрачка относительно уголков глаз. Позволяет заметить взгляд в сторону _без_ поворота головы.
3.  **Логика Анализа Поведения**:
    - **Машина Состояний**: Отслеживает статусы (`NORMAL`, `SUSPICIOUS`, `ALERT`, `CHEATING`) с временными задержками (например, отвод взгляда >3 секунд запускает тревогу).
    - **Запись Доказательств**: Автоматически записывает короткие видеоклипы (на стороне сервера), когда подтверждено нарушение.
4.  **Real-Time WebSocket API**: Низкие задержки передачи данных между фронтендом и Python-бэкендом.

### Использованные Датасеты

- **Roboflow Universe**: [Cellphone Computer Vision Model](https://universe.roboflow.com/d1156414/cellphone-0aodn) - Датасет "Phone on hand" (оптимизированный v4-v11) для детекции телефонов под сложными углами.
- **COCO128**: Стандартный датасет, отфильтрованный по классу `67` (Cell Phone) для поддержания базовой точности.
- **FPI (Face Phone Interaction)**: [Dataset from arXiv:2509.09111v1](https://arxiv.org/html/2509.09111v1) - Использовался для анализа специфических взаимодействий лица и телефона.

### Установка

1.  **Склонируйте репозиторий**:

    ```bash
    git clone <repo_url>
    cd phone_detecter
    ```

2.  **Настройте Виртуальное Окружение**:

    ```bash
    python -m venv venv
    ./venv/Scripts/Activate.ps1
    ```

3.  **Установите Зависимости**:

    ```bash
    pip install -r requirements.txt
    ```

4.  **Настройте Переменные Окружения** (для обучения):
    - Создайте файл `.env` или установите в системе: `ROBOFLOW_API_KEY=your_key`

### Использование

1.  **Запустите Сервер**:

    ```powershell
    ./run_server.bat
    ```

    - Или вручную: `uvicorn app.api.endpoints:app --reload`

2.  **Откройте Веб-Приложение**:
    - Перейдите по адресу `http://localhost:8000`.
