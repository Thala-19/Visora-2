import numpy as np
import cv2
import os
from ultralytics import YOLO
from pathlib import Path

# Threshold
conf_thr = 0.5

# Open Webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
cap.set(cv2.CAP_PROP_BRIGHTNESS, 100)

# Load Model (YOLO12n)
weights = Path("yolo12n.pt")
if not weights.exists():
    print(f"⚠️ ERROR: File model tidak ditemukan: {weights}")
    raise SystemExit

model = YOLO(str(weights))
names = model.names
Colors = np.random.uniform(0, 255, size=(len(names), 3))

# Set untuk menyimpan objek yang sudah terdeteksi
detected_objects = set()

# Output file
output_filename = "output.txt"
output_file = open(output_filename, "w", encoding="utf-8")
output_file.write("You are currently seeing \n")

while True:
    success, img = cap.read()
    if not success:
        print("⚠️ ERROR: Kamera tidak terbaca")
        break

    # YOLO inference
    results = model.predict(source=img, imgsz=640, conf=conf_thr, verbose=False)
    img_disp = img.copy()

    labels_in_frame = set()
    if results and len(results) > 0:
        r = results[0]
        if r.boxes is not None and len(r.boxes) > 0:
            for box, score, cls in zip(r.boxes.xyxy.cpu().numpy(),
                                       r.boxes.conf.cpu().numpy(),
                                       r.boxes.cls.cpu().numpy()):
                x1, y1, x2, y2 = box.astype(int)
                cls_id = int(cls)
                label = names.get(cls_id, str(cls_id))
                color = Colors[cls_id % len(Colors)]
                cv2.rectangle(img_disp, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img_disp, f"{label} {score:.2f}", (x1, max(0, y1-10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                labels_in_frame.add(label)

                if label not in detected_objects:
                    detected_objects.add(label)
                    output_file.write(label + "\n" + "then\n")
                    output_file.flush()
                    print(f"💾 Disimpan ke file: {label}")

    cv2.imshow("Output", img_disp)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
output_file.close()
cap.release()
cv2.destroyAllWindows()

print(f"✅ Selesai! Cek file: {output_filename}")
