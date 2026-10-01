import collections
import csv

from fleet_check import verify

truth = {r["image"]: r["correct_plate"].replace(" ", "").upper()
         for r in csv.DictReader(open("data/val_truth.csv")) if r["correct_plate"].strip()}
preds = {}
for r in csv.DictReader(open("runs/predictions.csv")):
    text = r["plate"].replace(" ", "")
    preds[r["image"]] = None if text in ("COULDNOTREAD", "NOPLATEFOUND") else text

APPROVED = {"MATCH", "PROBABLE_MATCH"}
genuine, impostor = collections.Counter(), collections.Counter()
plates = sorted(set(truth.values()))
for image, plate in truth.items():
    read = preds.get(image)
    genuine[verify(read, plate)[0]] += 1
    for other in plates:
        if other != plate:
            impostor[verify(read, other)[0]] += 1


def show(title, counts):
    total = sum(counts.values())
    print(f"\n{title} ({total} checks)")
    for decision in ("MATCH", "PROBABLE_MATCH", "MANUAL_CHECK", "MISMATCH"):
        print(f"  {decision:15} {counts[decision]:6}  {counts[decision] / total:6.1%}")
    return total


g = show("Genuine: card used on its own car", genuine)
i = show("Impostor: card used on a different car", impostor)
print(f"\nGenuine auto-approved:        {sum(genuine[d] for d in APPROVED) / g:.1%}  (higher is better)")
print(f"Genuine sent to manual check: {genuine['MANUAL_CHECK'] / g:.1%}")
print(f"Genuine wrongly flagged:      {genuine['MISMATCH'] / g:.1%}  (lower is better)")
print(f"Impostor auto-approved:       {sum(impostor[d] for d in APPROVED) / i:.2%}  (must be near 0)")
