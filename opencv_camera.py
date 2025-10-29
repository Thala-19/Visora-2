import numpy as np
import cv2
import os
from ultralytics import YOLO
from pathlib import Path

def main():
    # Threshold
    conf_thr = 0.5

    # Open Webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 100)

    # Load YOLO12n
    weights = Path("yolo12n.pt")
    if not weights.exists():
        print(f"⚠️ ERROR: Model file not found: {weights}")
        return

    model = YOLO(str(weights))
    names = model.names
    Colors = np.random.uniform(0, 255, size=(len(names), 3))

    # Output file
    output_filename = "output.txt"
    output_file = open(output_filename, "w", encoding="utf-8")
    output_file.write("You are currently seeing \n")
    detected_objects = set()

    while True:
        if os.path.exists("stop_camera.txt"):
            print("🛑 Stop signal received, exiting camera loop.")
            break

        success, img = cap.read()
        if not success:
            print("⚠️ ERROR: Kamera tidak terbaca")
            break

        # Inference
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

        for label in labels_in_frame:
            if label not in detected_objects:
                detected_objects.add(label)

        cv2.imshow("Output", img_disp)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Write unique labels to file
    object_list = list(detected_objects)
    for i, obj in enumerate(object_list):
        output_file.write(obj)
        if i != len(object_list) - 1:
            output_file.write("\nthen\n")
            print(f"💾 Disimpan ke file: {obj}")

    output_file.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
