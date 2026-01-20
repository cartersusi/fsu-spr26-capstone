from ultralytics import YOLO

from ultralytics import YOLO

def main():
    model = YOLO('yolov8n.pt')
    results = model.train(
        data='coco.yaml',
        epochs=100,
        imgsz=320,
        batch=16,
        device=0,
        project='nano_vehicle',
        name='nano_vehicle'
    )

if __name__ == '__main__':
    main()