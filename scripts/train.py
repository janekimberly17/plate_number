from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8n.pt")          # small pretrained model, downloads automatically
    model.train(
        data="configs/data.yaml",
        imgsz=1024,      # your images are 1024x1024
        epochs=50,       # passes over the training set
        batch=8,         # lower to 4 if you get "CUDA out of memory"
        fliplr=0.0,      # never mirror images: a flipped plate isn't a real plate
        patience=20,     # stop early if it stops improving
        name="plate_v8n",
        device=0,        # use the GPU
    )