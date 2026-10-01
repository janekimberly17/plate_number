import csv

truth = {r["image"]: r["correct_plate"].replace(" ", "").upper()
         for r in csv.DictReader(open("data/val_truth.csv"))
         if r["correct_plate"].strip()}
preds = {r["image"]: r for r in csv.DictReader(open("runs/predictions.csv"))}

right, wrong = 0, []
for image, answer in truth.items():
    p = preds.get(image)
    got = p["plate"].replace(" ", "").upper() if p else "MISSING"
    if got == answer:
        right += 1
    else:
        wrong.append((image, got, p["raw_ocr"] if p else "", answer))

print(f"Plate accuracy: {right}/{len(truth)} = {right / len(truth):.1%}")
print("\nWrong reads (predicted | raw OCR | truth):")
for image, got, raw, answer in wrong:
    print(f"  {got:15} | {raw:20} | {answer}   {image[:20]}")