"""Step 1: find out what the downloaded dataset actually contains.

    python scripts/inspect_dataset.py data

Prints image counts, the annotation format it detects (YOLO txt, Pascal VOC
xml, COCO json, CSV), the classes used, and boxes per image.
"""
from __future__ import annotations

import collections
import json
import statistics
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

IMG = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main(root: Path) -> None:
    files = [p for p in root.rglob("*") if p.is_file()]
    by_ext = collections.Counter(p.suffix.lower() for p in files)
    images = [p for p in files if p.suffix.lower() in IMG]
    print(f"{len(files)} files under {root}")
    print("extensions:", dict(by_ext.most_common()))
    print("top-level dirs:", sorted({p.relative_to(root).parts[0] for p in files if len(p.relative_to(root).parts) > 1}))

    classes, per_image, fmt = collections.Counter(), [], None
    for xml in [p for p in files if p.suffix.lower() == ".xml"]:
        try:
            ann = ET.parse(xml).getroot()
        except ET.ParseError:
            continue
        if ann.tag != "annotation":
            continue
        fmt = "pascal_voc"
        objs = ann.findall("object")
        per_image.append(len(objs))
        classes.update(o.findtext("name", "").strip() for o in objs)

    if fmt is None:
        for txt in [p for p in files if p.suffix.lower() == ".txt" and p.name != "classes.txt"]:
            rows = [r.split() for r in txt.read_text(errors="ignore").splitlines() if r.strip()]
            # 5 numbers = box; more (odd count >= 7) = polygon outline. YOLO accepts both.
            if rows and all(r[0].isdigit() and (len(r) == 5 or (len(r) >= 7 and len(r) % 2 == 1)) for r in rows):
                fmt = "yolo (boxes)" if all(len(r) == 5 for r in rows) and fmt != "yolo (polygons)" else "yolo (polygons)"
                per_image.append(len(rows))
                classes.update(r[0] for r in rows)

    if fmt is None:
        for js in [p for p in files if p.suffix.lower() == ".json"]:
            try:
                data = json.loads(js.read_text())
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if isinstance(data, dict) and {"images", "annotations", "categories"} <= data.keys():
                fmt = "coco"
                cats = {c["id"]: c["name"] for c in data["categories"]}
                classes.update(cats[a["category_id"]] for a in data["annotations"])
                counts = collections.Counter(a["image_id"] for a in data["annotations"])
                per_image.extend(counts.values())

    csvs = [p for p in files if p.suffix.lower() == ".csv"]
    print(f"\n{len(images)} images; annotation format: {fmt or 'none detected'}")
    if fmt is None:
        sample = next((p for p in files if p.suffix.lower() == ".txt"), None)
        if sample:
            print(f"  first line of {sample.name}: {sample.read_text(errors='ignore').splitlines()[:1]}")
    if csvs:
        print("CSV files (may hold plate text):", [str(p.relative_to(root)) for p in csvs[:5]])
        print("  first lines of", csvs[0].name, ":", csvs[0].read_text(errors="ignore").splitlines()[:3])
    if per_image:
        print(f"boxes per labelled image: mean {statistics.mean(per_image):.1f}, max {max(per_image)}")
        print(f"{len(classes)} classes, most common:", classes.most_common(40))
        if len(classes) > 20 or statistics.mean(per_image) > 3:
            print("-> looks CHARACTER-level: you can train a 2nd YOLO for characters")
        else:
            print("-> looks PLATE-level: train the detector, then use OCR on the crop")
    if images:
        try:
            import cv2
            sizes = collections.Counter()
            for p in images[:200]:
                im = cv2.imread(str(p))
                if im is not None:
                    sizes[f"{im.shape[1]}x{im.shape[0]}"] += 1
            print("image sizes (first 200):", sizes.most_common(5))
        except ImportError:
            pass


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "data"))