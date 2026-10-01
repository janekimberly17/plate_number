"""Run plate reading on every validation image and save the results to a CSV.

    python scripts/batch_predict.py            # default settings
    python scripts/batch_predict.py 1.33       # try a different WIDEN value
"""
import csv
import sys
from pathlib import Path

import cv2
import easyocr
from ultralytics import YOLO

from plate_format import normalize_boxes

WEIGHTS = "models/plate_v8n.pt"  # copy of runs/detect/plate_v8n/weights/best.pt
IMAGES = Path("data/val/images")
PLATE_CLASS = 1
ALLOWED = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
# The dataset images were stretched to 1024x1024, which squeezes characters
# sideways (W and M start to look like H). Stretching the crop back helps.
DEFAULT_WIDEN = 1.33


def read_plate(img, model, reader, widen=DEFAULT_WIDEN):
    """Find the plate in a BGR image and read it.

    Returns (plate or None, detector confidence, raw OCR text, crop image).
    """
    result = model.predict(img, imgsz=1024, conf=0.4, verbose=False)[0]
    plates = [b for b in result.boxes if int(b.cls) == PLATE_CLASS]
    if not plates:
        return None, None, "", None
    # The car at the pump is the closest one, so its plate is the biggest box.
    best = max(plates, key=lambda b: float((b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1])))
    x1, y1, x2, y2 = map(int, best.xyxy[0].tolist())
    pad_x, pad_y = int(0.08 * (x2 - x1)), int(0.15 * (y2 - y1))
    crop = img[max(0, y1 - pad_y):y2 + pad_y, max(0, x1 - pad_x):x2 + pad_x]
    crop = cv2.resize(crop, None, fx=2 * widen, fy=2, interpolation=cv2.INTER_CUBIC)

    found = reader.readtext(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), allowlist=ALLOWED)
    boxes = [(min(p[0] for p in q), min(p[1] for p in q), max(p[1] for p in q) - min(p[1] for p in q), t)
             for q, t, c in found]
    plate, raw = normalize_boxes(boxes)
    return plate, float(best.conf), raw, crop


if __name__ == "__main__":
    widen = float(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_WIDEN
    model = YOLO(WEIGHTS)
    reader = easyocr.Reader(["en"], gpu=True)
    crops_dir = Path("runs/crops")
    crops_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(IMAGES.glob("*.jpg")):
        plate, det_conf, raw, crop = read_plate(cv2.imread(str(path)), model, reader, widen)
        if crop is None:
            rows.append([path.name, "", "", "NO PLATE FOUND"])
            continue
        cv2.imwrite(str(crops_dir / path.name), crop)
        rows.append([path.name, f"{det_conf:.2f}", raw, plate.display if plate else "COULD NOT READ"])
        print(f"{path.name[:40]:40} {rows[-1][3]}")

    with open("runs/predictions.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "detector_conf", "raw_ocr", "plate"])
        writer.writerows(rows)
    read = sum(1 for r in rows if r[3] not in ("NO PLATE FOUND", "COULD NOT READ"))
    print(f"\nWIDEN={widen}: read a valid plate in {read}/{len(rows)} images. Now run: python scripts/score.py")