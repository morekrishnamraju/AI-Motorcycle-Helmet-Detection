from modules.config import MODEL_PATH
from modules.detector import ULTRALYTICS_AVAILABLE
if not ULTRALYTICS_AVAILABLE:
    print('Ultralytics not available')
else:
    from ultralytics import YOLO
    model = YOLO(str(MODEL_PATH))
    print('Model path:', MODEL_PATH)
    print('Class names mapping (id -> name):')
    print(model.names)
