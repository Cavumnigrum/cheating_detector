const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const resultArea = document.getElementById('result-area');
const sourceImage = document.getElementById('source-image');
const detectionCanvas = document.getElementById('detection-canvas');
const loader = document.getElementById('loader');

// Элементы переключения режимов
const btnUpload = document.getElementById('btn-upload');
const btnWebcam = document.getElementById('btn-webcam');
const uploadSection = document.getElementById('upload-section');
const webcamSection = document.getElementById('webcam-section');

// Элементы веб-камеры
const webcamVideo = document.getElementById('webcam-video');
const webcamCanvas = document.getElementById('webcam-canvas'); // Используется для отрисовки наложений
const btnStartCam = document.getElementById('btn-start-cam');
const btnCalibrate = document.getElementById('btn-calibrate');
const btnRecord = document.getElementById('btn-record');
const logsList = document.getElementById('logs-list');

let ws = null;
let mediaRecorder = null;
let recordedChunks = [];
let isRecording = false;

// --- Переключение режимов ---
btnUpload.addEventListener('click', () => {
    setActiveMode('upload');
    stopWebcam();
});

btnWebcam.addEventListener('click', () => {
    setActiveMode('webcam');
});

function setActiveMode(mode) {
    if (mode === 'upload') {
        btnUpload.classList.add('active');
        btnWebcam.classList.remove('active');
        uploadSection.classList.remove('hidden');
        webcamSection.classList.add('hidden');
    } else {
        btnUpload.classList.remove('active');
        btnWebcam.classList.add('active');
        uploadSection.classList.add('hidden');
        webcamSection.classList.remove('hidden');
    }
}

// --- Реализация загрузки ---
dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('dragover'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', (e) => { if (e.target.files.length) handleFile(e.target.files[0]); });

function handleFile(file) {
    if (!file.type.startsWith('image/')) { alert('Please upload an image file'); return; }
    resultArea.classList.add('hidden');
    loader.classList.remove('hidden');
    const reader = new FileReader();
    reader.onload = (e) => {
        sourceImage.src = e.target.result;
        sourceImage.onload = () => uploadAndDetect(file);
    };
    reader.readAsDataURL(file);
}

async function uploadAndDetect(file) {
    const formData = new FormData();
    formData.append('file', file);
    try {
        const response = await fetch('/api/detect', { method: 'POST', body: formData });
        if (!response.ok) throw new Error('Detection failed');
        const data = await response.json();
        drawDetections(detectionCanvas, sourceImage, data.detections);
    } catch (error) {
        console.error(error);
        alert('Error detecting phones.');
    } finally {
        loader.classList.add('hidden');
        resultArea.classList.remove('hidden');
    }
}

// --- Реализация веб-камеры ---
btnStartCam.addEventListener('click', startWebcam);

async function startWebcam() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' } });
        webcamVideo.srcObject = stream;
        await webcamVideo.play();
        
        btnStartCam.innerText = "Stop Camera";
        btnStartCam.onclick = stopWebcam;
        
        btnCalibrate.disabled = false; // Включить калибровку
        
        // Запуск WS
        connectWebSocket();
        
    } catch (err) {
        console.error("Error accessing webcam:", err);
        alert("Could not access webcam.");
    }
}

function stopWebcam() {
    if (webcamVideo.srcObject) {
        webcamVideo.srcObject.getTracks().forEach(track => track.stop());
        webcamVideo.srcObject = null;
    }
    if (streamInterval) clearInterval(streamInterval);
    if (ws) ws.close();
    
    // Очистка холста
    const ctx = webcamCanvas.getContext('2d');
    ctx.clearRect(0, 0, webcamCanvas.width, webcamCanvas.height);
    
    btnStartCam.innerText = "Start Camera";
    btnStartCam.onclick = startWebcam;
}

function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${protocol}//${window.location.host}/api/ws/detect`);
    
    ws.onopen = () => {
        console.log("WS Connected");
        startStreaming();
    };
    
    ws.onmessage = (event) => {
        const response = JSON.parse(event.data);
        drawWebcamDetections(response.detections, response.behavior); // Теперь используем 'behavior'
        updateSessionLog(response.behavior.history); // Новая панель логов
    };
    
    ws.onclose = () => console.log("WS Closed");
}

btnCalibrate.addEventListener('click', () => {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "calibrate" }));
        
        // Визуальная обратная связь
        const originalText = btnCalibrate.innerText;
        btnCalibrate.innerText = "Calibrated!";
        btnCalibrate.classList.remove("secondary");
        btnCalibrate.classList.add("active");
        
        setTimeout(() => {
            btnCalibrate.innerText = "Recalibrate";
            btnCalibrate.classList.add("secondary");
            btnCalibrate.classList.remove("active");
        }, 2000);
    }
});

// --- Реализация записи ---
btnRecord.addEventListener('click', () => {
    toggleRecording();
});

function toggleRecording() {
    if (isRecording) {
        stopRecording();
    } else {
        startRecording();
    }
}

function startRecording() {
    if (webcamVideo.srcObject) {
         try {
             // Запись холста, а не видео, чтобы включить наложения!
             // Или видео? Требование - "умная запись видео".
             // Захват потока Canvas лучше для доказательств (включает рамки).
             const stream = webcamCanvas.captureStream(30); 
             mediaRecorder = new MediaRecorder(stream, { mimeType: 'video/webm' });
             
             recordedChunks = [];
             mediaRecorder.ondataavailable = (event) => {
                 if (event.data.size > 0) recordedChunks.push(event.data);
             };
             
             mediaRecorder.onstop = saveRecording;
             
             mediaRecorder.start();
             isRecording = true;
             btnRecord.innerText = "Stop Recording";
             btnRecord.classList.add("recording"); // Добавить стиль красной пульсации, если нужно
             console.log("Recording started...");
         } catch (e) {
             console.error("Recording failed", e);
             alert("Could not start recording");
         }
    }
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
        mediaRecorder.stop();
        isRecording = false;
        btnRecord.innerText = "Start Recording";
        btnRecord.classList.remove("recording");
        console.log("Recording stopped...");
    }
}

function saveRecording() {
    const blob = new Blob(recordedChunks, { type: 'video/webm' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    document.body.appendChild(a);
    a.style = 'display: none';
    a.href = url;
    a.download = `evidence_${new Date().toISOString().replace(/:/g, '-')}.webm`;
    a.click();
    window.URL.revokeObjectURL(url);
    console.log("Recording saved");
}

function updateSessionLog(history) {
    // Логика перемещена сюда для чистоты
    if (!history) return;
    
    logsList.innerHTML = '';
    [...history].reverse().forEach(evt => {
        const div = document.createElement('div');
        let severityClass = 'alert-green';
        if (evt.severity > 50) severityClass = 'alert-orange';
        if (evt.severity > 80) severityClass = 'alert-red';
        
        div.className = `log-entry ${severityClass}`;
        const t = new Date(evt.timestamp * 1000).toLocaleTimeString();
        div.innerHTML = `<span>${evt.code}</span> <span>${t}</span>`;
        logsList.appendChild(div);
    });
}

let streamInterval = null;

function startStreaming() {
    if (streamInterval) clearInterval(streamInterval);
    
    // Создание закадрового холста для захвата кадров без влияния на UI
    const captureCanvas = document.createElement('canvas');
    const captureCtx = captureCanvas.getContext('2d');
    
    // Контроль FPS (отправка каждые 100мс = 10fps, или 33мс = 30fps)
    // Слишком быстрая отправка может перегрузить WS, если бэкенд медленный.
    streamInterval = setInterval(() => {
        if (!ws || ws.readyState !== WebSocket.OPEN) return;
        
        // 1. Отрисовка кадра видео на закадровый холст
        if (webcamVideo.readyState === webcamVideo.HAVE_ENOUGH_DATA) {
             captureCanvas.width = webcamVideo.videoWidth;
             captureCanvas.height = webcamVideo.videoHeight;
             captureCtx.drawImage(webcamVideo, 0, 0, captureCanvas.width, captureCanvas.height);
             
             // 2. Конвертация в Blob/Buffer
             captureCanvas.toBlob((blob) => {
                 if (blob) {
                     // Проверка, открыт ли WS (асинхронно)
                     if (ws.readyState === WebSocket.OPEN) {
                        ws.send(blob);
                     }
                 }
             }, 'image/jpeg', 0.8); // 80% качество JPEG
        }
    }, 100); // 10 FPS
}

function drawWebcamDetections(detections, behavior) {
    if (webcamCanvas.width !== webcamVideo.videoWidth) {
        webcamCanvas.width = webcamVideo.videoWidth;
        webcamCanvas.height = webcamVideo.videoHeight;
    }
    
    const ctx = webcamCanvas.getContext('2d');
    ctx.clearRect(0, 0, webcamCanvas.width, webcamCanvas.height);
    
    // Отрисовка статуса / предупреждений
    drawBehaviorStatus(ctx, behavior);

    // Отрисовка обнаруженных телефонов
    if (detections && detections.length > 0) {
        drawBoxes(ctx, detections);
    }
    
    // Отрисовка сетки лица
    if (behavior.landmarks) {
        ctx.fillStyle = '#00ffaa'; // Голубовато-зеленый
        behavior.landmarks.forEach(lm => {
            const x = lm.x * webcamCanvas.width;
            const y = lm.y * webcamCanvas.height;
            ctx.beginPath();
            ctx.arc(x, y, 1, 0, 2 * Math.PI);
            ctx.fill();
        });
    }
}

function drawBehaviorStatus(ctx, behavior) {
    if (!behavior) return;
    
    const state = behavior.state;
    const score = behavior.score;
    const message = behavior.message;
    
    let color = "#22c55e"; // Зеленый (Норма)
    
    if (state === "SUSPICIOUS") color = "#f59e0b"; // Оранжевый
    if (state === "CHEATING_SUSPECTED" || state === "FACE_MISSING" || state === "ALERT" || state === "CHEATING") color = "#ef4444"; // Красный
    
    // Верхняя панель
    ctx.fillStyle = "rgba(15, 23, 42, 0.9)";
    ctx.fillRect(0, 0, 450, 100);
    
    ctx.fillStyle = color;
    ctx.font = 'bold 28px Outfit';
    ctx.fillText(`${state} (${score}%)`, 20, 40);
    
    ctx.fillStyle = "#cbd5e1";
    ctx.font = '18px Outfit';
    ctx.fillText(message || "Monitoring behavior...", 20, 70);
    
    // Отрисовка лога истории (Слева внизу)
    // УДАЛЕНО: Перемещено в updateSessionLog, вызываемое из ws.onmessage
    // чтобы отделить логику UI от логики отрисовки Canvas.

    // --- ЛОГИКА АВТО-ЗАПИСИ ---
    // ОТКЛЮЧЕНО: Логика теперь обрабатывается на 100% на сервере в logic.py
    // Фронтенд должен просто отображать статус.
    /*
    const critical = behavior.state === "CHEATING_SUSPECTED" || behavior.state === "SUSPICIOUS" || behavior.state === "ALERT";
    
    if (critical && !isRecording) {
        console.log("Auto-starting recording due to behavior alert");
        toggleRecording(); // Start
    }
    
    if (!critical && isRecording && behavior.state === "NORMAL") {
        if (!window.stopTimeout) {
             window.stopTimeout = setTimeout(() => {
                 if (isRecording) {
                     console.log("Auto-stopping recording (Normal state restored)");
                     toggleRecording(); // Stop
                 }
                 window.stopTimeout = null;
             }, 3000); // 3 seconds buffer
        }
    } else {
        if (window.stopTimeout && critical) {
            clearTimeout(window.stopTimeout);
            window.stopTimeout = null;
        }
    }
    */
}

// Общая логика отрисовки
function drawDetections(canvas, imgElement, detections) {
    canvas.width = imgElement.naturalWidth;
    canvas.height = imgElement.naturalHeight;
    canvas.style.width = '100%';
    canvas.style.height = '100%';
    
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    drawBoxes(ctx, detections);
}

function drawBoxes(ctx, detections) {
    ctx.lineWidth = 4;
    ctx.font = 'bold 24px Outfit';

    detections.forEach(det => {
        const [x1, y1, x2, y2] = det.bbox;
        // Использовать красный для обнаружения обмана
        const color = '#ef4444'; // Красный-500
        const labelText = det.label || `Phone ${(det.conf * 100).toFixed(1)}%`;
        const score = ` ${(det.conf * 100).toFixed(0)}%`;

        // Рамка
        ctx.strokeStyle = color;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

        // Фон метки
        const fullText = labelText + score;
        const textWidth = ctx.measureText(fullText).width;
        
        ctx.fillStyle = color;
        ctx.fillRect(x1, y1 - 34, textWidth + 20, 34);

        // Текст метки
        ctx.fillStyle = '#ffffff';
        ctx.fillText(fullText, x1 + 10, y1 - 8);
    });
}
