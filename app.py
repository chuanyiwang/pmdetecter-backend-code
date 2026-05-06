# =============================================================================
# app.py  --  Clean Flask REST + SocketIO server
#
# Main frontend endpoints:
#   GET /latest       -> latest flat JSON for Flutter frontend
#   GET /incidents    -> incident history generated from AI detection + PM10
#
# Other REST endpoints:
#   GET  /api/status
#   GET  /api/reading/latest
#   GET  /api/reading/history
#   GET  /api/spikes
#   GET  /api/spikes/latest
#   GET  /api/detections
#   GET  /api/snapshot
#   POST /api/config
#
# WebSocket namespace:
#   /ws
#
# This cleaned version removes:
#   - coral_detector.py dependency
#   - correlation.py dependency
#   - CoralCameraManager
#   - PM10Monitor
#
# AI detection is read from standalone ai_server.py:
#   http://127.0.0.1:5001/ai/detection
# =============================================================================

import time
import logging
import threading
import requests

from collections import deque
from datetime import datetime, timezone

from flask import Flask, jsonify, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS

import config
from pm_reader import PMSensorManager
from spike_detector import SpikeDetector


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("app")


# ---------------------------------------------------------------------------
# Flask + SocketIO setup
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.config["SECRET_KEY"] = config.SECRET_KEY
CORS(app, origins=config.CORS_ORIGINS)

socketio = SocketIO(
    app,
    cors_allowed_origins=config.CORS_ORIGINS,
    async_mode=config.SOCKETIO_ASYNC,
)


# ---------------------------------------------------------------------------
# Component initialisation
# ---------------------------------------------------------------------------
sensor_mgr = PMSensorManager()
spike_det = SpikeDetector()


# ---------------------------------------------------------------------------
# Global caches
# ---------------------------------------------------------------------------
_detection_incidents = []

_last_spike_ts = None
_last_processed_reading_ts = None

_last_detection_label = None
_last_incident_time = 0.0

# AI server from ai_server.py
AI_DETECTION_URL = "http://127.0.0.1:5001/ai/detection"

# Incident debounce
INCIDENT_COOLDOWN_S = 5.0


# ---------------------------------------------------------------------------
# REST helpers
# ---------------------------------------------------------------------------
def _ok(data):
    return jsonify({"status": "ok", "data": data})


def _err(msg, code=400):
    return jsonify({"status": "error", "message": msg}), code


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def _safe_int(value, default=0):
    try:
        if value is None:
            return default
        return int(float(value))
    except (ValueError, TypeError):
        return default


def _safe_float(value, default=None):
    try:
        if value is None:
            return default
        return float(value)
    except (ValueError, TypeError):
        return default


def _estimate_pm_channels(pm10):
    """
    Fallback estimation if only PM10 is available.
    Ensures PM1 <= PM2.5 <= PM4 <= PM10.
    """
    pm10 = max(0, _safe_int(pm10, 0))
    pm4 = max(0, pm10 - 10)
    pm25 = max(0, pm4 - 8)
    pm1 = max(0, pm25 - 6)

    return pm1, pm25, pm4, pm10


def _normalize_object_label(label):
    """
    Normalize labels from ai_server.py.

    Raw AI labels:
        heavy vehicles
        light_vehicles
        smoke
        two-wheelers
        None
    """
    if not label:
        return "Uncertain"

    text = str(label).strip().lower()
    text = text.replace("_", " ")
    text = text.replace("-", " ")

    if text in ["none", "unknown", "uncertain", ""]:
        return "Uncertain"

    heavy_keywords = [
        "heavy vehicle",
        "heavy vehicles",
        "truck",
        "lorry",
        "bus",
        "van",
        "pickup",
    ]

    light_keywords = [
        "light vehicle",
        "light vehicles",
        "car",
        "sedan",
        "hatchback",
        "suv",
    ]

    two_wheeler_keywords = [
        "two wheeler",
        "two wheelers",
        "bike",
        "motorbike",
        "motorcycle",
    ]

    smoke_keywords = [
        "smoke",
        "vapour",
        "vapor",
    ]

    if any(k in text for k in heavy_keywords):
        return "heavy vehicles"

    if any(k in text for k in light_keywords):
        return "light_vehicles"

    if any(k in text for k in two_wheeler_keywords):
        return "two-wheelers"

    if any(k in text for k in smoke_keywords):
        return "smoke"

    return "Uncertain"


def _get_ai_detection_from_ai_server():
    """
    Read latest AI detection result from ai_server.py.

    Expected response:
    {
        "object": "light_vehicles",
        "confidence": 0.823,
        "timestamp": "12:34:56"
    }
    """
    try:
        resp = requests.get(AI_DETECTION_URL, timeout=0.5)

        if resp.status_code != 200:
            return {
                "raw_object": None,
                "object": "Uncertain",
                "confidence": 0.0,
                "timestamp": "",
                "available": False,
            }

        data = resp.json()

        raw_object = data.get("object")
        confidence = _safe_float(data.get("confidence"), 0.0)
        ai_timestamp = str(data.get("timestamp", ""))

        normalized = _normalize_object_label(raw_object)

        return {
            "raw_object": raw_object,
            "object": normalized,
            "confidence": confidence,
            "timestamp": ai_timestamp,
            "available": True,
        }

    except Exception as exc:
        log.warning("Failed to read AI detection from ai_server: %s", exc)

        return {
            "raw_object": None,
            "object": "Uncertain",
            "confidence": 0.0,
            "timestamp": "",
            "available": False,
        }


def _is_valid_object_label(label):
    if not label:
        return False

    text = str(label).strip().lower()
    return text not in ["uncertain", "unknown", "none", ""]


def _append_detection_incident(object_label, reading):
    """
    Save one incident using:
        - normalized AI object label
        - current corrected PM10 reading
    """
    global _detection_incidents

    if not _is_valid_object_label(object_label):
        return

    timestamp = datetime.now(timezone.utc).isoformat()
    pm10_value = 0.0

    if isinstance(reading, dict):
        timestamp = str(reading.get("timestamp", timestamp))
        pm10_value = round(_safe_float(reading.get("pm10"), 0.0), 2)

    incident = {
        "time": timestamp,
        "pm10": pm10_value,
        "object": str(object_label),
    }

    _detection_incidents.insert(0, incident)

    if len(_detection_incidents) > 50:
        _detection_incidents = _detection_incidents[:50]


def _flatten_latest_for_flutter(reading):
    """
    Convert full sensor reading into flat JSON expected by Flutter frontend.
    """
    if not reading:
        return None

    # PM10
    pm10 = reading.get("pm10")
    if pm10 is None:
        pm10 = reading.get("pm_10")
    if pm10 is None:
        pm10 = reading.get("value")
    if pm10 is None:
        pm10 = reading.get("pm")

    pm10 = _safe_float(pm10, 0.0)

    # PM1 / PM2.5 / PM4
    pm1 = reading.get("pm1")
    pm25 = reading.get("pm25")
    pm4 = reading.get("pm4")

    if pm25 is None:
        pm25 = reading.get("pm2_5")
    if pm25 is None:
        pm25 = reading.get("pm2.5")

    if pm1 is None or pm25 is None or pm4 is None:
        est_pm1, est_pm25, est_pm4, est_pm10 = _estimate_pm_channels(pm10)

        pm1 = est_pm1 if pm1 is None else _safe_float(pm1, float(est_pm1))
        pm25 = est_pm25 if pm25 is None else _safe_float(pm25, float(est_pm25))
        pm4 = est_pm4 if pm4 is None else _safe_float(pm4, float(est_pm4))
        pm10 = float(est_pm10)
    else:
        pm1 = _safe_float(pm1, 0.0)
        pm25 = _safe_float(pm25, 0.0)
        pm4 = _safe_float(pm4, 0.0)

    # Temperature / humidity
    temperature = reading.get("temperature")
    if temperature is None:
        temperature = reading.get("temp")

    humidity = reading.get("humidity")
    if humidity is None:
        humidity = reading.get("rh")

    temperature = _safe_float(temperature, None)
    humidity = _safe_float(humidity, None)

    # AI object
    ai = _get_ai_detection_from_ai_server()
    object_label = ai.get("object", "Uncertain")

    timestamp = reading.get("timestamp")
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat()

    return {
        "pm1": round(float(pm1), 2),
        "pm25": round(float(pm25), 2),
        "pm4": round(float(pm4), 2),
        "pm10": round(float(pm10), 2),
        "temperature": None if temperature is None else round(float(temperature), 2),
        "humidity": None if humidity is None else round(float(humidity), 2),
        "object": str(object_label),
        "timestamp": str(timestamp),
    }


# ---------------------------------------------------------------------------
# Background processing loop
# ---------------------------------------------------------------------------
def _push_loop():
    """
    Main background loop:
      1. Get latest sensor reading.
      2. Update spike detector once per new reading.
      3. Read AI detection from ai_server.py.
      4. Generate incidents from valid AI object + current PM10.
      5. Emit WebSocket events.
    """
    global _last_spike_ts
    global _last_processed_reading_ts
    global _last_detection_label
    global _last_incident_time

    while True:
        time.sleep(config.EMIT_INTERVAL_S)

        try:
            reading = sensor_mgr.get_latest()

            if not reading:
                continue

            timestamp = reading.get("timestamp")
            pm10_value = _safe_float(reading.get("pm10"), None)

            spike = None

            # ---------------------------------------------------------------
            # Spike detection
            # Only process each sensor timestamp once.
            # ---------------------------------------------------------------
            if (
                timestamp is not None
                and timestamp != _last_processed_reading_ts
                and pm10_value is not None
            ):
                _last_processed_reading_ts = timestamp
                spike = spike_det.update(pm10_value, timestamp)

            # ---------------------------------------------------------------
            # AI detection
            # ---------------------------------------------------------------
            ai = _get_ai_detection_from_ai_server()
            object_label = ai.get("object", "Uncertain")
            confidence = _safe_float(ai.get("confidence"), 0.0)

            # ---------------------------------------------------------------
            # Incident generation
            # ---------------------------------------------------------------
            now = time.time()
            cooldown_finished = (now - _last_incident_time) >= INCIDENT_COOLDOWN_S

            if (
                _is_valid_object_label(object_label)
                and object_label != _last_detection_label
                and cooldown_finished
            ):
                _append_detection_incident(object_label, reading)

                _last_detection_label = object_label
                _last_incident_time = now

                log.info(
                    "Incident recorded: object=%s confidence=%.2f pm10=%s",
                    object_label,
                    confidence,
                    reading.get("pm10"),
                )

            # ---------------------------------------------------------------
            # WebSocket reading push
            # ---------------------------------------------------------------
            payload = {
                **reading,
                "stats": spike_det.get_stats(),
                "ai_detection": ai,
            }
            socketio.emit("reading", payload, namespace="/ws")

            # ---------------------------------------------------------------
            # WebSocket AI detection push
            # ---------------------------------------------------------------
            socketio.emit("detection", ai, namespace="/ws")

            # ---------------------------------------------------------------
            # WebSocket spike push
            # ---------------------------------------------------------------
            if spike and spike.get("timestamp") != _last_spike_ts:
                _last_spike_ts = spike.get("timestamp")
                socketio.emit("spike", spike, namespace="/ws")

                log.info(
                    "Spike emitted: direction=%s severity=%s pm10=%s",
                    spike.get("direction"),
                    spike.get("severity"),
                    spike.get("pm10_value"),
                )

        except Exception as exc:
            log.error("Push loop error: %s", exc)


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------
@app.route("/api/status", methods=["GET"])
def status():
    return _ok({
        "service": "PM Monitor",
        "version": "2.0.0-clean",
        "server_time": datetime.now(timezone.utc).isoformat(),
        "mock_sensor": config.MOCK_SENSOR,
        "spike_cfg": {
            "window": config.SPIKE_WINDOW,
            "z_threshold": config.SPIKE_Z_THRESHOLD,
            "min_delta": config.SPIKE_MIN_DELTA,
        },
        "ai_detection_url": AI_DETECTION_URL,
    })


@app.route("/api/reading/latest", methods=["GET"])
def reading_latest():
    reading = sensor_mgr.get_latest()

    if not reading:
        return _err("No readings available yet", 503)

    stats = spike_det.get_stats()
    ai = _get_ai_detection_from_ai_server()

    return _ok({
        **reading,
        "stats": stats,
        "ai_detection": ai,
    })


@app.route("/api/reading/history", methods=["GET"])
def reading_history():
    try:
        n = int(request.args.get("n", 50))
        n = max(1, min(n, config.SPIKE_HISTORY_MAX))
    except ValueError:
        return _err("n must be an integer")

    history = sensor_mgr.get_history(n)

    return _ok({
        "count": len(history),
        "readings": history,
    })


@app.route("/api/spikes", methods=["GET"])
def spikes_all():
    try:
        last = int(request.args.get("last", 100))
    except ValueError:
        return _err("last must be an integer")

    spikes = spike_det.get_spikes(last_n=last)

    return _ok({
        "count": len(spikes),
        "spikes": spikes,
    })


@app.route("/api/spikes/latest", methods=["GET"])
def spike_latest():
    all_spikes = spike_det.get_spikes()
    spike = all_spikes[-1] if all_spikes else None

    return _ok(spike)


@app.route("/api/detections", methods=["GET"])
def detections():
    """
    Debug endpoint for latest AI detection from ai_server.py.
    """
    ai = _get_ai_detection_from_ai_server()
    return _ok(ai)


@app.route("/api/snapshot", methods=["GET"])
def snapshot():
    reading = sensor_mgr.get_latest()
    spikes = spike_det.get_spikes()
    stats = spike_det.get_stats()
    ai = _get_ai_detection_from_ai_server()

    return _ok({
        "reading": reading,
        "stats": stats,
        "latest_spike": spikes[-1] if spikes else None,
        "spike_count": len(spikes),
        "ai_detection": ai,
        "incident_count": len(_detection_incidents),
        "server_time": datetime.now(timezone.utc).isoformat(),
    })


@app.route("/api/config", methods=["POST"])
def update_config():
    body = request.get_json(silent=True) or {}

    if "z_threshold" in body:
        config.SPIKE_Z_THRESHOLD = float(body["z_threshold"])

    if "min_delta" in body:
        config.SPIKE_MIN_DELTA = float(body["min_delta"])

    if "window" in body:
        new_w = int(body["window"])
        config.SPIKE_WINDOW = new_w
        spike_det._window = deque(spike_det._window, maxlen=new_w)

    return _ok({
        "z_threshold": config.SPIKE_Z_THRESHOLD,
        "min_delta": config.SPIKE_MIN_DELTA,
        "window": config.SPIKE_WINDOW,
    })


# ---------------------------------------------------------------------------
# Frontend endpoints
# ---------------------------------------------------------------------------
@app.route("/latest", methods=["GET"])
def latest_for_flutter():
    reading = sensor_mgr.get_latest()

    if not reading:
        return jsonify(None), 200

    payload = _flatten_latest_for_flutter(reading)
    return jsonify(payload), 200


@app.route("/incidents", methods=["GET"])
def incidents_for_flutter():
    """
    Return incident history for Flutter.
    Each incident combines:
      - current corrected PM10
      - AI object label from ai_server.py
    """
    return jsonify(_detection_incidents), 200


# ---------------------------------------------------------------------------
# WebSocket events
# ---------------------------------------------------------------------------
@socketio.on("connect", namespace="/ws")
def on_connect():
    log.info("WebSocket client connected")
    emit("status", {"message": "Connected to PM monitor"})


@socketio.on("disconnect", namespace="/ws")
def on_disconnect():
    log.info("WebSocket client disconnected")


@socketio.on("request_snapshot", namespace="/ws")
def on_request_snapshot():
    reading = sensor_mgr.get_latest()
    spikes = spike_det.get_spikes()
    stats = spike_det.get_stats()
    ai = _get_ai_detection_from_ai_server()

    emit("snapshot", {
        "reading": reading,
        "stats": stats,
        "latest_spike": spikes[-1] if spikes else None,
        "ai_detection": ai,
        "incident_count": len(_detection_incidents),
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    log.info("Starting clean PM monitoring backend...")

    # Start PM / temperature / humidity sensor manager
    sensor_mgr.start()

    # Start background processing loop
    push_thread = threading.Thread(target=_push_loop, daemon=True)
    push_thread.start()

    log.info(
        "Flask listening on http://%s:%d",
        config.FLASK_HOST,
        config.FLASK_PORT,
    )

    log.info(
        "Reading AI detection from %s",
        AI_DETECTION_URL,
    )

    socketio.run(
        app,
        host=config.FLASK_HOST,
        port=config.FLASK_PORT,
        debug=config.FLASK_DEBUG,
        use_reloader=False,
        allow_unsafe_werkzeug=True,
    )