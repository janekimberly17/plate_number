import csv
import json
import shutil
from pathlib import Path

import cv2
import easyocr
from ultralytics import YOLO

from batch_predict import IMAGES, PLATE_CLASS, WEIGHTS, read_plate

OUT = Path("web/public/demo")
MAX_WIDTH = 640  # keeps the website small and fast
COLORS = {0: (235, 160, 40), PLATE_CLASS: (60, 200, 60)}  # BGR: car blue-ish, plate green

if __name__ == "__main__":
    truth = {r["image"]: r["correct_plate"].replace(" ", "").upper()
             for r in csv.DictReader(open("data/val_truth.csv")) if r["correct_plate"].strip()}
    shutil.rmtree(OUT, ignore_errors=True)
    (OUT / "images").mkdir(parents=True)
    (OUT / "crops").mkdir()

    model = YOLO(WEIGHTS)
    reader = easyocr.Reader(["en"], gpu=True)
    samples = []
    for n, path in enumerate(sorted(p for p in IMAGES.glob("*.jpg") if p.name in truth), 1):
        img = cv2.imread(str(path))
        plate, det_conf, raw, crop = read_plate(img, model, reader)

        # Draw every detected car and plate box.
        drawn = img.copy()
        for box in model.predict(img, imgsz=1024, conf=0.4, verbose=False)[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cv2.rectangle(drawn, (x1, y1), (x2, y2), COLORS.get(int(box.cls), (200, 200, 200)), 4)
        scale = MAX_WIDTH / drawn.shape[1]
        drawn = cv2.resize(drawn, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        sid = f"v{n:03d}"
        cv2.imwrite(str(OUT / "images" / f"{sid}.jpg"), drawn, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if crop is not None:
            cv2.imwrite(str(OUT / "crops" / f"{sid}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
        samples.append({
            "id": sid,
            "truth": truth[path.name],
            "read": plate.text if plate else None,
            "display": plate.display if plate else None,
            "raw": raw,
            "detConf": round(det_conf, 2) if det_conf is not None else None,
            "hasCrop": crop is not None,
        })
        print(f"{sid}  truth {truth[path.name]:10}  read {plate.display if plate else '-'}")

    # Demo fleet: one card per distinct plate, with made-up drivers.
    plates = sorted({s["truth"] for s in samples})
    fleet = [{"cardId": f"CARD-{i:03d}", "plate": p, "driver": f"Driver {i}"} for i, p in enumerate(plates, 1)]
    (OUT / "results.json").write_text(json.dumps({"samples": samples, "fleet": fleet}, indent=1))
    print(f"\nExported {len(samples)} samples and {len(fleet)} cards to {OUT}/")
