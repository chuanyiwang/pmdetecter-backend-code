# =============================================================================
# spike_detector.py  --  Detects statistically significant PM10 spikes
#
# Algorithm:
#   1. Maintain a rolling window of the last N readings.
#   2. Compute mean and std of that window.
#   3. A new reading is a spike if:
#        |reading - mean| / std  >  SPIKE_Z_THRESHOLD
#        AND  |reading - mean|   >  SPIKE_MIN_DELTA  (absolute guard)
#   4. Direction: RISING if reading > mean, FALLING otherwise.
# =============================================================================

import math
import logging
from datetime import datetime
from collections import deque

import config

log = logging.getLogger(__name__)


class SpikeDetector:
    """
    Stateful spike detector.  Feed it one PM10 value at a time via update().
    Call get_spikes() to retrieve the event log.
    """

    def __init__(self):
        self._window = deque(maxlen=config.SPIKE_WINDOW)
        self._spikes = []            # list of spike event dicts

    # ------------------------------------------------------------------ API
    def update(self, pm10_value, timestamp=None):
        """
        Evaluate one new pm10 sample.

        Returns a spike dict if a spike is detected, else None.
        The reading is always appended to the rolling window afterwards.
        """
        ts = timestamp or (datetime.utcnow().isoformat() + "Z")
        result = None

        if len(self._window) >= config.SPIKE_WINDOW:
            mean, std = self._stats()
            if std > 0:
                z_score = (pm10_value - mean) / std
                delta   = pm10_value - mean

                if (abs(z_score) >= config.SPIKE_Z_THRESHOLD and
                        abs(delta) >= config.SPIKE_MIN_DELTA):

                    direction = "RISING"  if delta > 0 else "FALLING"
                    severity  = self._severity(abs(z_score))

                    spike = {
                        "timestamp":   ts,
                        "pm10_value":  round(pm10_value, 2),
                        "window_mean": round(mean, 2),
                        "window_std":  round(std, 2),
                        "z_score":     round(z_score, 3),
                        "delta":       round(delta, 2),
                        "direction":   direction,
                        "severity":    severity,
                        "source":      None,   # filled in by CorrelationEngine
                    }
                    self._spikes.append(spike)
                    log.info(
                        "SPIKE %s  pm10=%.1f  z=%.2f  severity=%s",
                        direction, pm10_value, z_score, severity
                    )
                    result = spike

        self._window.append(pm10_value)
        return result

    def get_spikes(self, last_n=None):
        """Return the spike event list (optionally trimmed to last_n)."""
        return self._spikes if last_n is None else self._spikes[-last_n:]

    def get_stats(self):
        """Return current window statistics."""
        if len(self._window) < 2:
            return {"mean": None, "std": None, "window_size": len(self._window)}
        m, s = self._stats()
        return {
            "mean":        round(m, 2),
            "std":         round(s, 2),
            "window_size": len(self._window),
        }

    def clear_spikes(self):
        self._spikes.clear()

    # ------------------------------------------------------------- internals
    def _stats(self):
        values = list(self._window)
        n      = len(values)
        mean   = sum(values) / n
        var    = sum((v - mean) ** 2 for v in values) / (n - 1)
        std    = math.sqrt(var)
        return mean, std

    @staticmethod
    def _severity(z_score):
        """Map z-score magnitude to a human-readable severity label."""
        if z_score >= 5.0:
            return "CRITICAL"
        elif z_score >= 4.0:
            return "HIGH"
        elif z_score >= 3.0:
            return "MEDIUM"
        else:
            return "LOW"
