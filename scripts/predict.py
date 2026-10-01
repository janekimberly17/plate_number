import sys

import cv2
import easyocr
from ultralytics import YOLO

from plate_format import join_lines, normalize

WEIGHTS = "runs/detect/plate_v8n/weights/best.pt"
PLATE_CLASS = 1  # 0 = car, 1 = car plate

if __name__ == "__main__":
    image_path = r"C:\Users\AIC\Documents\plate_number\data\val\images\3d7577dd-9427-45c0-92f3-e8f35543346a_jpg.rf.766932d0861da0352e719b41a0a3211e.jpg"
    model = YOLO(WEIGHTS)
    reader = easyocr.Reader(["en"], gpu=True)

    # 1. Detect: keep only plate boxes, take the most confident one.
    img = cv2.imread(image_path)
    result = model.predict(img, imgsz=1024, conf=0.4, verbose=False)[0]
    plates = [b for b in result.boxes if int(b.cls) == PLATE_CLASS]
    if not plates:
        sys.exit("No plate found")
    best = max(plates, key=lambda b: float(b.conf))
    x1, y1, x2, y2 = map(int, best.xyxy[0].tolist())

    # 2. Crop with a little padding and enlarge so the characters are clearer.
    pad = int(0.08 * (x2 - x1))
    crop = img[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad]
    crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # 3. OCR: only letters and digits can appear on a plate.
    found = reader.readtext(gray, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    boxes = [(min(p[0] for p in quad), min(p[1] for p in quad),
              max(p[1] for p in quad) - min(p[1] for p in quad), text)
             for quad, text, conf in found]
    raw = join_lines(boxes)  # handles two-row plates
    ocr_conf = sum(c for _, _, c in found) / len(found) if found else 0.0

    # 4. Clean up look-alike mistakes using Malaysian plate rules.
    plate = normalize(raw)
    print(f"detector confidence: {float(best.conf):.2f}")
    print(f"raw OCR text:        {raw!r} (confidence {ocr_conf:.2f})")
    print(f"cleaned plate:       {plate.display if plate else 'could not read'}")

    cv2.imwrite("runs/last_crop.jpg", crop)
    print("crop saved to runs/last_crop.jpg")
