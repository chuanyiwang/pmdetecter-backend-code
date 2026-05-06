#!/bin/bash

LOG_FILE="/home/pi/dashboard_autostart.log"

echo "==============================" >> "$LOG_FILE"
echo "Script started at $(date)" >> "$LOG_FILE"

#sleep 10

echo "Checking WiFi connection..." >> "$LOG_FILE"

# Check whether WiFi is connected to an SSID
if ! iwgetid -r > /dev/null 2>&1; then
    echo "WiFi is not connected. Browser not opened." >> "$LOG_FILE"
    exit 0
fi

echo "WiFi is connected: $(iwgetid -r)" >> "$LOG_FILE"
echo "Checking local frontend service..." >> "$LOG_FILE"

for i in {1..40}; do
    if curl -s --max-time 2 http://127.0.0.1:8000 > /dev/null; then
        echo "Frontend ready. Opening Chromium at $(date)" >> "$LOG_FILE"
        DISPLAY=:0 XAUTHORITY=/home/pi/.Xauthority /usr/bin/chromium --kiosk --incognito http://127.0.0.1:8000
        exit 0
    fi

    echo "Frontend not ready, retry $i" >> "$LOG_FILE"
    sleep 1
done

echo "Frontend not ready after waiting. Browser not opened." >> "$LOG_FILE"
exit 0
