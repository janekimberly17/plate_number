import csv
import sys
from pathlib import Path

# Characters OCR mixes up on Malaysian plates. Swapping within a group is cheap.
LOOKALIKE_GROUPS = ["HWMN", "7ZT1", "0QDO", "8B3", "5S", "2Z", "6G", "VY", "1IL", "CG"]
LOOKALIKE_COST = 0.3
PROBABLE_LIMIT = 0.6   # up to two look-alike swaps
MANUAL_LIMIT = 1.6     # anything further apart is a different plate


def _swap_cost(a, b):
    if a == b:
        return 0.0
    if any(a in g and b in g for g in LOOKALIKE_GROUPS):
        return LOOKALIKE_COST
    return 1.0


def plate_distance(a, b):
    """Edit distance where look-alike swaps cost less than real differences."""
    prev = [float(j) for j in range(len(b) + 1)]
    for i, ca in enumerate(a, 1):
        cur = [float(i)]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + _swap_cost(ca, cb)))
        prev = cur
    return prev[-1]


def verify(read_plate, expected_plate):
    """Compare the plate text the camera read with the card's plate.

    read_plate: plate text without spaces, or None if nothing could be read.
    Returns (decision, distance).
    """
    if not read_plate:
        return "MANUAL_CHECK", None
    d = plate_distance(read_plate.replace(" ", ""), expected_plate.replace(" ", "").upper())
    if d == 0:
        return "MATCH", d
    if d <= PROBABLE_LIMIT:
        return "PROBABLE_MATCH", d
    if d <= MANUAL_LIMIT:
        return "MANUAL_CHECK", d
    return "MISMATCH", d


def load_fleet(path="data/fleet.csv"):
    with open(path, newline="") as f:
        return {row["card_id"]: row for row in csv.DictReader(f)}


if __name__ == "__main__":
    import cv2
    import easyocr
    from ultralytics import YOLO

    from batch_predict import WEIGHTS, read_plate

    if len(sys.argv) != 3:
        sys.exit("usage: python scripts/fleet_check.py <image> <card_id>")
    image_path, card_id = sys.argv[1], sys.argv[2]
    if not Path(image_path).exists():  # allow the start of a filename, e.g. data/val/images/20251123_140020
        matches = sorted(Path(image_path).parent.glob(Path(image_path).name + "*"))
        if not matches:
            sys.exit(f"No image found starting with {image_path}")
        image_path = str(matches[0])
    if not Path("data/fleet.csv").exists():
        sys.exit("data/fleet.csv is missing. Copy it from the zip into your data folder.")
    fleet = load_fleet()
    if card_id not in fleet:
        sys.exit(f"Unknown card {card_id}. Cards in fleet.csv: {', '.join(fleet)}")
    card = fleet[card_id]

    plate, det_conf, raw, _ = read_plate(cv2.imread(image_path), YOLO(WEIGHTS), easyocr.Reader(["en"], gpu=True))
    decision, distance = verify(plate.text if plate else None, card["plate"])

    print(f"Card:          {card_id} ({card['driver']}, assigned to {card['plate']})")
    print(f"Camera read:   {plate.display if plate else 'nothing readable'}   (raw OCR {raw!r})")
    print(f"Decision:      {decision}" + (f"   (distance {distance:.1f})" if distance is not None else ""))
