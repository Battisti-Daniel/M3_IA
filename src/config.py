from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
DEFAULT_LOG_FILE = LOGS_DIR / "detections.csv"
DEFAULT_TIMEOUT_SECONDS = 4.0
DEFAULT_TARGET_FPS = 20

# Configurações de confiança mínima para detecções
MIN_PLATE_CONFIDENCE = 0.50  # Confiança mínima YOLO para placa (50%)
MIN_OCR_CONFIDENCE = 0.30    # Confiança mínima OCR (30%)
MIN_PLATE_LENGTH = 3         # Tamanho mínimo da placa (ex: ABC1234)
MAX_PLATE_LENGTH = 8         # Tamanho máximo da placa (ex: ABC1D234)

# Otimizações de FPS
YOLO_IMGSZ = 640              # Tamanho da imagem para YOLO (menor = mais rápido: 320, 416, 640)
OCR_SKIP_FRAMES = 3           # Rodar OCR a cada N frames (1 = sempre, 3 = a cada 3 frames)
OCR_CONFIDENCE_THRESHOLD = 0.85  # Se OCR > esse valor, não re-processar a placa
PLATE_SKIP_IF_CONFIDENT = True   # Pular detecção de placa se já tiver leitura confiável

# Configurações de visualização
SHOW_VIDEO_OVERLAY = True         # Mostrar bboxes, texto e info no vídeo
SHOW_FPS_OVERLAY = True           # Mostrar FPS no canto do vídeo
SHOW_VEHICLE_BBOX = True          # Mostrar bbox do veículo
SHOW_PLATE_TEXT = True            # Mostrar texto da placa no vídeo
SHOW_CONFIDENCE = True            # Mostrar porcentagem de confiança

# Modelos padrão
DEFAULT_VEHICLE_MODEL = "yolo11s.pt"  # YOLO11s - Detecta veículos (car, motorcycle, etc)
DEFAULT_PLATE_MODEL = "license-plate-v1s.pt"  # HuggingFace YOLO11s fine-tuned para placas

# Modelos disponíveis:
# ==================
# DETECÇÃO DE VEÍCULOS (YOLO11 pré-treinado no COCO):
#   - yolo11n.pt  (5.6MB)  - Nano: mais rápido, menos preciso
#   - yolo11s.pt  (19MB)   - Small: equilíbrio velocidade/precisão (RECOMENDADO)
#   - yolo11m.pt  (40MB)   - Medium: mais preciso, mais lento
#   - yolo11l.pt  (51MB)   - Large: alta precisão, mais lento
#
# DETECÇÃO DE PLACAS (fine-tuned para license plates):
#   - license-plate-v1n.pt (5.4MB)  - Nano: mais rápido
#   - license-plate-v1s.pt (19MB)   - Small: equilíbrio (RECOMENDADO)
#   - license-plate-v1x.pt (114MB)  - Extra-Large: máxima precisão
#   - license_plate_detector.pt (5.4MB) - Alternativo (bhaskrr/21k imagens)
#   - nosso_modelo_yolo11n.pt (5.6MB) - Modelo treinado localmente

VIDEO_SOURCES = {
    "Webcam 0": 0,
    "Webcam 1": 1,
}


for directory in (MODELS_DIR, LOGS_DIR):
    directory.mkdir(parents=True, exist_ok=True)
