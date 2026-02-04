import argparse
import cv2
import os
from tqdm import tqdm
from ultralytics import YOLO

DEFAULT_SIMULATION_DASHCAM = "./support/simulation/dashcam.mp4"
DEFAULT_MODEL_PATH = "./model/nano_vehicle_320/weights/best.pt"

model = None


def load_model(model_path=DEFAULT_MODEL_PATH):
    """Load the YOLO model for object detection."""
    global model
    if model is None:
        model = YOLO(model_path)
        print(f"Loaded model from {model_path}")
    return model


def detect_objects(frame, model_path=DEFAULT_MODEL_PATH, conf_threshold=0.25, imgsz=640):
    """
    Run object detection on a frame and draw bounding boxes.
    
    Args:
        frame: The input image/frame (numpy array from cv2)
        model_path: Path to the YOLO model weights
        conf_threshold: Confidence threshold for detections
        imgsz: Inference size (YOLO letterboxes internally)
    
    Returns:
        annotated_frame: Frame with bounding boxes drawn
        detections: List of detection dictionaries
    """
    yolo_model = load_model(model_path)
    
    results = yolo_model(frame, conf=conf_threshold, imgsz=imgsz, verbose=False)
    
    detections = []
    annotated_frame = frame.copy()
    
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
            
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            confidence = float(box.conf[0].cpu().numpy())
            class_id = int(box.cls[0].cpu().numpy())
            class_name = yolo_model.names[class_id]
            
            detections.append({
                'class_name': class_name,
                'confidence': confidence,
                'bbox': (x1, y1, x2, y2)
            })
            
            color = (0, 255, 0)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            label = f"{class_name}: {confidence:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            cv2.rectangle(annotated_frame, 
                         (x1, y1 - text_height - 10), 
                         (x1 + text_width + 5, y1), 
                         color, -1)
            cv2.putText(annotated_frame, label, 
                       (x1 + 2, y1 - 5), 
                       font, font_scale, (0, 0, 0), thickness)
    
    return annotated_frame, detections


def process_frame(frame, frame_index, output_dir="frames", imgsz=640, run_detection=True):
    """
    Process a single frame: run detection and save at original resolution.
    """
    detections = []
    annotated_frame = frame
    
    if run_detection:
        annotated_frame, detections = detect_objects(frame, imgsz=imgsz)
    
    frame_path = os.path.join(output_dir, f"frame_{frame_index:05d}.jpg")
    cv2.imwrite(frame_path, annotated_frame)
    
    return frame_path, detections


def extract_frames(video_path, output_dir="frames", fps=10, imgsz=640, run_detection=True):
    """
    Extract frames from video, optionally running object detection on each frame.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(video_fps / fps))
    
    print(f"Video: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ {video_fps:.1f}fps")
    print(f"Inference size: {imgsz} (YOLO will letterbox internally)")
    
    frame_paths = []
    all_detections = []
    frame_count = 0
    saved_count = 0
    
    pbar = tqdm(total=total_frames, desc="Processing frames")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            frame_path, detections = process_frame(
                frame, saved_count, output_dir, imgsz, run_detection
            )
            frame_paths.append(frame_path)
            all_detections.append(detections)
            
            if detections:
                det_summary = ", ".join([f"{d['class_name']}({d['confidence']:.2f})" for d in detections])
                tqdm.write(f"Frame {saved_count}: {det_summary}")
            
            saved_count += 1
        
        frame_count += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()
    
    print(f"Extracted {saved_count} frames to {output_dir}")
    if run_detection:
        total_detections = sum(len(d) for d in all_detections)
        print(f"Total detections: {total_detections}")
    
    return frame_paths, all_detections


def main(dashcam_footage: str, run_detection: bool = True, imgsz: int = 640):
    frame_paths, detections = extract_frames(dashcam_footage, imgsz=imgsz, run_detection=run_detection)
    return frame_paths, detections


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dashcam frame processing with object detection.")
    parser.add_argument('--input', type=str, default=DEFAULT_SIMULATION_DASHCAM,
                       help=f"Input video path (default: {DEFAULT_SIMULATION_DASHCAM})")
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL_PATH,
                       help=f"Model weights path (default: {DEFAULT_MODEL_PATH})")
    parser.add_argument('--no-detection', action='store_true',
                       help="Disable object detection")
    parser.add_argument('--output', type=str, default="frames",
                       help="Output directory for frames (default: frames)")
    parser.add_argument('--fps', type=int, default=10,
                       help="Frames per second to extract (default: 10)")
    parser.add_argument('--imgsz', type=int, default=640,
                       help="YOLO inference size (default: 640)")
    
    args = parser.parse_args()
    
    if args.model != DEFAULT_MODEL_PATH:
        model = None
        DEFAULT_MODEL_PATH = args.model
    
    main(args.input, run_detection=not args.no_detection, imgsz=args.imgsz)