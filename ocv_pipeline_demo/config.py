"""Configuration for the Vision AI Pipeline."""

# --- Paths ---
YOLO_MODEL_PATH = "/home/yolo26x.onnx"
YOLO_COMPILED_PATH = "/home/yolo26x_compiled.mxr"
QWEN_MODEL_PATH = "/models/Qwen3-VL-8B-Instruct"

# --- YOLO Preprocessing ---
INPUT_SIZE = (640, 640)  # (width, height)
CONF_THRESHOLD = 0.5
NMS_IOU_THRESHOLD = 0.45

# --- GPU ---
GPU_DEVICE_ID = 0

# --- vLLM / Qwen3-VL ---
VLLM_BASE_URL = "http://localhost:8198/v1"
VLLM_MODEL_NAME = "/models/Qwen3-VL-8B-Instruct"
VLM_MAX_TOKENS = 100
VLM_TOP_K_ROIS = 3  # max ROIs to send to VLM per frame
VLM_PROMPT = "Describe this object or scene in one concise sentence."

# --- COCO class names (80 classes) ---
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]
