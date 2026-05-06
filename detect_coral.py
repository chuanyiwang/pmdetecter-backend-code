import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import subprocess
import time
import threading
import signal
import sys

from ai_server import set_detection, set_frame, run_server

MODEL_PATH = "/home/pi/best_int8_edgetpu.tflite"
LABELS = ["heavy vehicles", "light_vehicles", "smoke", "two-wheelers"]
THRESHOLD = 0.4
COLORS = [(0,255,0),(0,0,255),(255,0,0),(0,255,255)]
INFER_EVERY = 3
WIDTH, HEIGHT = 640, 480

# Load Coral
delegate = tflite.load_delegate("libedgetpu.so.1")
interp = tflite.Interpreter(
    model_path=MODEL_PATH,
    experimental_delegates=[delegate]
)
interp.allocate_tensors()
inp = interp.get_input_details()
out = interp.get_output_details()
in_scale, in_zp = inp[0]["quantization"]
out_scale, out_zp = out[0]["quantization"]
print("Coral loaded!")

# Start camera
cam_proc = subprocess.Popen(
    ['rpicam-vid', '-t', '0',
     '--width', str(WIDTH), '--height', str(HEIGHT),
     '--codec', 'yuv420', '--nopreview',
     '--framerate', '30', '-o', '-'],
    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
)
print("Camera started!")

# Safe shutdown - prevents camera from getting locked
def cleanup(signum=None, frame=None):
    print("Shutting down safely...")
    try:
        cam_proc.terminate()
        cam_proc.wait(timeout=5)
    except Exception:
        cam_proc.kill()
    cv2.destroyAllWindows()
    print("Camera released. Done.")
    sys.exit(0)

signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)

# Start API server in background
threading.Thread(target=run_server, daemon=True).start()
print("AI server running:")
print("  Video stream: http://172.20.10.3:5001/ai/stream")
print("  Detection:    http://172.20.10.3:5001/ai/detection")

frame_size = WIDTH * HEIGHT * 3 // 2
frame_count = 0
last_boxes = []
fps_time = time.time()

try:
    while True:
        raw = cam_proc.stdout.read(frame_size)
        if len(raw) != frame_size:
            continue

        yuv = np.frombuffer(raw, dtype=np.uint8).reshape((HEIGHT * 3 // 2, WIDTH))
        frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
        frame_count += 1

        if frame_count % INFER_EVERY == 0:
            orig_h, orig_w = frame.shape[:2]
            img = cv2.resize(frame, (320, 320))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_float = img.astype(np.float32) / 255.0
            data = np.expand_dims(
                (img_float / in_scale + in_zp).astype(np.int8), 0
            )

            interp.set_tensor(inp[0]["index"], data)
            interp.invoke()

            raw_out = interp.get_tensor(out[0]["index"])[0]
            output = (raw_out.astype(np.float32) - out_zp) * out_scale
            boxes = output[:4, :]
            scores = output[4:, :]

            class_ids = np.argmax(scores, axis=0)
            confidences = scores[class_ids, np.arange(scores.shape[1])]
            mask = confidences > THRESHOLD

            raw_boxes = []
            raw_scores = []
            raw_classes = []

            if mask.any():
                good_idx = np.where(mask)[0]
                good_conf = confidences[mask]
                good_class = class_ids[mask]
                good_boxes = boxes[:, good_idx]

                cx, cy = good_boxes[0], good_boxes[1]
                bw, bh = good_boxes[2], good_boxes[3]

                x1s = ((cx - bw/2) * orig_w).astype(int)
                y1s = ((cy - bh/2) * orig_h).astype(int)
                x2s = ((cx + bw/2) * orig_w).astype(int)
                y2s = ((cy + bh/2) * orig_h).astype(int)

                for j in range(len(good_idx)):
                    raw_boxes.append([x1s[j], y1s[j], x2s[j], y2s[j]])
                    raw_scores.append(float(good_conf[j]))
                    raw_classes.append(int(good_class[j]))

            last_boxes = []
            if raw_boxes:
                indices = cv2.dnn.NMSBoxes(
                    raw_boxes, raw_scores, THRESHOLD, 0.5
                )
                for idx in indices:
                    x1,y1,x2,y2 = raw_boxes[idx]
                    label = LABELS[raw_classes[idx]]
                    conf = raw_scores[idx]
                    last_boxes.append((x1,y1,x2,y2,label,conf))

                best = max(last_boxes, key=lambda x: x[5])
                set_detection(best[4], best[5])
            else:
                set_detection("None", 0.0)

        # Draw detection boxes
        for (x1,y1,x2,y2,label,conf) in last_boxes:
            color = COLORS[LABELS.index(label)]
            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
            cv2.putText(frame,f"{label} {conf:.2f}",(x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)

        # Send frame to video stream API
        ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ret:
            set_frame(jpeg.tobytes())

        # FPS counter
        if frame_count % 30 == 0:
            now = time.time()
            fps = 30 / (now - fps_time)
            fps_time = now
            print(f"FPS: {fps:.1f}")

except Exception as e:
    print(f"Error: {e}")
finally:
    cleanup()
