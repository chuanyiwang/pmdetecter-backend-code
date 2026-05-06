# =============================================================================
# config.py  --  Central configuration for PM10 + Coral AI monitoring system
# =============================================================================

# --- Sensor ---
SENSOR_PORT        = "/dev/ttyUSB0"   # SDS011 serial port on RPi 5
SENSOR_BAUD        = 9600
SENSOR_READ_HZ     = 1.0              # readings per second
MOCK_SENSOR        = False            # set True for bench-testing without hardware

# --- Spike detection ---
SPIKE_WINDOW       = 30               # rolling window length (samples)
SPIKE_Z_THRESHOLD  = 2.5             # z-score threshold to flag a spike
SPIKE_MIN_DELTA    = 10.0            # minimum absolute delta (ug/m3) to count
SPIKE_HISTORY_MAX  = 500             # max readings kept in memory

# --- Coral AI camera ---
MOCK_CORAL         = False            # set True to run without Edge TPU
MODEL_PATH         = "models/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite"
LABEL_PATH         = "models/coco_labels.txt"
CAMERA_INDEX       = 0
CORAL_CONFIDENCE   = 0.45            # minimum detection confidence
FRAME_INTERVAL_S   = 2.0            # seconds between coral inference calls

# --- Pollution source mapping ---
# Maps COCO object labels to pollution source categories
SOURCE_MAP = {
    "car":          "Traffic",
    "truck":        "Traffic",
    "bus":          "Traffic",
    "motorcycle":   "Traffic",
    "bicycle":      "Traffic",
    "person":       "Unknown / Pedestrian",
    "fire":         "Wood Fire / Bonfire",
    "smoke":        "Wood Fire / Bonfire",
    "train":        "Industrial / Heavy Traffic",
    "boat":         "Industrial / Vessel",
    "airplane":     "Industrial / Aviation",
    "backpack":     "Unknown",
    "suitcase":     "Unknown",
}
DEFAULT_SOURCE = "Unknown"

# --- Flask ---
FLASK_HOST         = "0.0.0.0"
FLASK_PORT         = 5000
FLASK_DEBUG        = False
SECRET_KEY         = "pm10-rpi5-secret"  # change in production
CORS_ORIGINS       = "*"                  # restrict in production

# --- WebSocket ---
SOCKETIO_ASYNC     = "threading"
EMIT_INTERVAL_S    = 1.0             # how often to push updates to connected clients
