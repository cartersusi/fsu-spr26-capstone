from ultralytics import YOLO


def main():
    model = YOLO("yolov8n.pt")

    results = model.train(
        data="bdd100k.yaml",
        workers=8,
        epochs=300,
        imgsz=640,
        batch=32,
        device=0,
        patience=50,
        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        cos_lr=True,
        augment=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        project="bdd100k_vehicle_640",
        name="run",
    )

    # model = YOLO('best.pt')
    # model.export(format='onnx', imgsz=640, simplify=True, opset=12)


if __name__ == "__main__":
    main()
