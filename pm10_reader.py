import time
import struct
import board
import adafruit_dht
import threading
import logging

from collections import deque
from datetime import datetime
from smbus2 import SMBus, i2c_msg

import config

log = logging.getLogger(__name__)


# =========================================================
# DHT11
# =========================================================
dht_sensor = adafruit_dht.DHT11(board.D4)


# =========================================================
# PMS5003
# =========================================================
"""
import serial

uart = serial.Serial("/dev/serial0", 9600, timeout=0.1)
pms_buffer = b""


def read_pms5003_once():
    global pms_buffer

    data = uart.read(128)
    if data:
        pms_buffer += data
    else:
        return None

    if len(pms_buffer) > 2048:
        pms_buffer = pms_buffer[-512:]

    while len(pms_buffer) >= 32:
        start = pms_buffer.find(b"\\x42\\x4d")

        if start == -1:
            pms_buffer = pms_buffer[-1:]
            return None

        if start > 0:
            pms_buffer = pms_buffer[start:]

        if len(pms_buffer) < 32:
            return None

        frame = pms_buffer[:32]

        frame_length = (frame[2] << 8) | frame[3]
        checksum_received = (frame[30] << 8) | frame[31]
        checksum_calculated = sum(frame[0:30])

        if frame_length != 28:
            pms_buffer = pms_buffer[1:]
            continue

        if checksum_received != checksum_calculated:
            pms_buffer = pms_buffer[1:]
            continue

        pms_buffer = pms_buffer[32:]

        pm1_0 = (frame[10] << 8) | frame[11]
        pm2_5 = (frame[12] << 8) | frame[13]
        pm10 = (frame[14] << 8) | frame[15]

        return pm1_0, pm2_5, pm10

    return None


def read_pms5003_wait(timeout_s=1.5):
    start_time = time.time()

    while time.time() - start_time < timeout_s:
        result = read_pms5003_once()
        if result is not None:
            return result
        time.sleep(0.02)

    return None
"""


# =========================================================
# SPS30
# =========================================================
I2C_BUS = 1
SPS30_ADDR = 0x69

CMD_START_MEASUREMENT = 0x0010
CMD_READ_DATA_READY = 0x0202
CMD_READ_MEASURED_VALUES = 0x0300
CMD_STOP_MEASUREMENT = 0x0104

bus = SMBus(I2C_BUS)


def crc8(data_bytes):
    crc = 0xFF
    for byte in data_bytes:
        crc ^= byte
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ 0x31) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc


def sps30_write_command(cmd, data_words=None):
    payload = [(cmd >> 8) & 0xFF, cmd & 0xFF]

    if data_words:
        for word in data_words:
            msb = (word >> 8) & 0xFF
            lsb = word & 0xFF
            payload.extend([msb, lsb, crc8([msb, lsb])])

    msg = i2c_msg.write(SPS30_ADDR, payload)
    bus.i2c_rdwr(msg)


def sps30_read_bytes(cmd, num_bytes):
    sps30_write_command(cmd)
    time.sleep(0.01)
    read = i2c_msg.read(SPS30_ADDR, num_bytes)
    bus.i2c_rdwr(read)
    return list(read)


def sps30_parse_with_crc(raw):
    data = []
    for i in range(0, len(raw), 3):
        b1, b2, crc = raw[i], raw[i + 1], raw[i + 2]
        if crc8([b1, b2]) != crc:
            raise ValueError("SPS30 CRC mismatch")
        data.extend([b1, b2])
    return bytes(data)


def sps30_start_measurement():
    sps30_write_command(CMD_START_MEASUREMENT, [0x0300])
    time.sleep(0.05)


def sps30_stop_measurement():
    sps30_write_command(CMD_STOP_MEASUREMENT)
    time.sleep(0.05)


def sps30_data_ready():
    raw = sps30_read_bytes(CMD_READ_DATA_READY, 3)
    parsed = sps30_parse_with_crc(raw)
    value = (parsed[0] << 8) | parsed[1]
    return value == 1


def sps30_read_measured_values():
    raw = sps30_read_bytes(CMD_READ_MEASURED_VALUES, 60)
    parsed = sps30_parse_with_crc(raw)
    values = struct.unpack(">10f", parsed)

    return {
        "pm1_0": values[0],
        "pm2_5": values[1],
        "pm4_0": values[2],
        "pm10": values[3],
    }


# =========================================================
# PM10SensorManager
# =========================================================
class PM10SensorManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._history = deque(maxlen=config.SPIKE_HISTORY_MAX)
        self._latest = None
        self._running = False
        self._thread = None

        self._last_sps = {
            "pm1_0": None,
            "pm2_5": None,
            "pm4_0": None,
            "pm10": None,
        }

        self._last_temperature = None
        self._last_humidity = None

        log.info("PM10SensorManager initialised with SPS30 + DHT11")

    def start(self):
        if self._running:
            return

        try:
            sps30_start_measurement()
            log.info("SPS30 measurement started")
        except Exception as exc:
            log.error("Failed to start SPS30: %s", exc)

        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        log.info("PM10SensorManager started")

    def stop(self):
        self._running = False

        try:
            sps30_stop_measurement()
        except Exception:
            pass

        try:
            bus.close()
        except Exception:
            pass

        try:
            dht_sensor.exit()
        except Exception:
            pass

    def get_latest(self):
        with self._lock:
            return dict(self._latest) if self._latest else None

    def get_history(self, n=None):
        with self._lock:
            data = list(self._history)
        return data if n is None else data[-n:]

    def _loop(self):
        interval = 1.0 / config.SENSOR_READ_HZ

        while self._running:
            t0 = time.time()

            reading = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "pm1": None,
                "pm25": None,
                "pm4": None,
                "pm10": None,
                "temperature": None,
                "humidity": None,
                "sps_pm1_0": None,
                "sps_pm25": None,
                "sps_pm4_0": None,
                "sps_pm10": None,
                "pm_source": "SPS30",
            }

            # -----------------------------
            # Read DHT11
            # Keep last valid values if one read fails
            # -----------------------------
            try:
                temp = dht_sensor.temperature
                hum = dht_sensor.humidity

                if temp is not None:
                    self._last_temperature = temp
                if hum is not None:
                    self._last_humidity = hum

            except Exception as exc:
                log.warning("DHT11 read failed: %s", exc)

            reading["temperature"] = self._last_temperature
            reading["humidity"] = self._last_humidity

            # -----------------------------
            # Read SPS30
            # -----------------------------
            try:
                if sps30_data_ready():
                    sps_values = sps30_read_measured_values()
                    self._last_sps["pm1_0"] = round(sps_values["pm1_0"], 2)
                    self._last_sps["pm2_5"] = round(sps_values["pm2_5"], 2)
                    self._last_sps["pm4_0"] = round(sps_values["pm4_0"], 2)
                    self._last_sps["pm10"] = round(sps_values["pm10"], 2)
            except Exception as exc:
                log.warning("SPS30 read failed: %s", exc)

            # -----------------------------
            # Always copy latest cached SPS30 values into reading
            # -----------------------------
            reading["sps_pm1_0"] = self._last_sps["pm1_0"]
            reading["sps_pm25"] = self._last_sps["pm2_5"]
            reading["sps_pm4_0"] = self._last_sps["pm4_0"]
            reading["sps_pm10"] = self._last_sps["pm10"]

            reading["pm1"] = self._last_sps["pm1_0"]
            reading["pm25"] = self._last_sps["pm2_5"]
            reading["pm4"] = self._last_sps["pm4_0"]
            reading["pm10"] = self._last_sps["pm10"]

            log.info(
                "PM loop values -> pm1=%s pm25=%s pm4=%s pm10=%s temp=%s hum=%s",
                reading["pm1"],
                reading["pm25"],
                reading["pm4"],
                reading["pm10"],
                reading["temperature"],
                reading["humidity"],
            )

            with self._lock:
                self._latest = reading
                self._history.append(reading)

            elapsed = time.time() - t0
            time.sleep(max(0.0, interval - elapsed))
