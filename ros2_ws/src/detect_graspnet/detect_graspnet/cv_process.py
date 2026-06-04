import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.models.sam import Predictor as SAMPredictor

import logging

logging.getLogger("ultralytics").setLevel(logging.WARNING)

def choose_model():
    """Initialize SAM predictor with proper parameters"""
    model_weight = 'sam_b.pt'
    overrides = dict(
        task='segment',
        mode='predict',
        # imgsz=1024,
        model=model_weight,
        conf=0.25,
        save=False
    )
    return SAMPredictor(overrides=overrides)


def detect_objects(image_or_path, target_class=None):
    """
    Detect objects with YOLO-World
    image_or_path: can be a file path (str) or a numpy array (image).
    Returns: (list of bboxes in xyxy format, visualization image)
    """
    model = YOLO("yolov8s-world.pt")
    if target_class:
        model.set_classes([target_class])

    results = model.predict(image_or_path, verbose=False)

    if not results or len(results) == 0:
        if isinstance(image_or_path, str):
            vis_img = cv2.imread(image_or_path)
        else:
            vis_img = image_or_path.copy()
        return [], vis_img

    result = results[0]
    boxes = result.boxes
    vis_img = result.plot()  # Get visualized detection results

    # Extract valid detections
    valid_boxes = []
    for box in boxes:
        if box.conf.item() > 0.10:  # Confidence threshold 0.10
            valid_boxes.append({
                "xyxy": box.xyxy[0].tolist(),
                "conf": box.conf.item(),
                "cls": result.names[int(box.cls.item())]
            })

    return valid_boxes, vis_img


def process_sam_results(results):
    """Process SAM results to get mask and center point"""
    if not results or not results[0].masks:
        return None, None

    mask = results[0].masks.data[0].cpu().numpy()
    mask = (mask > 0).astype(np.uint8) * 255

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    M = cv2.moments(contours[0])
    if M["m00"] == 0:
        return None, mask

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy), mask


def segment_image(image_path):
    """
    image_path: can be either a file path (str) or a numpy array (BGR image).
    Returns the segmentation mask as a numpy array, or None on failure.
    """
    target_class = "banana"
    detections, vis_img = detect_objects(image_path, target_class)

    if isinstance(image_path, str):
        bgr_img = cv2.imread(image_path)
        if bgr_img is None:
            raise ValueError(f"Failed to read image from path: {image_path}")
        image_rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB)

    predictor = choose_model()
    predictor.set_image(image_rgb)

    if detections:
        best_det = max(detections, key=lambda x: x["conf"])
        results = predictor(bboxes=[best_det["xyxy"]])
        center, mask = process_sam_results(results)
        print(f"Auto-selected {best_det['cls']} with confidence {best_det['conf']:.2f}")
    else:
        print("No detections - click on target object")
        cv2.imshow('Select Object', vis_img)

        point = []
        clicked = False
        def click_handler(event, x, y, flags, param):
            nonlocal clicked
            if event == cv2.EVENT_LBUTTONDOWN:
                print(f"Clicked at ({x}, {y})")
                point.extend([x, y])
                clicked = True
        cv2.setMouseCallback('Select Object', click_handler)
        print("Waiting for user click...")
        while not clicked:
            key = cv2.waitKey(10)
            if key == 27:
                raise ValueError("User cancelled selection")
        cv2.destroyAllWindows()

        if len(point) == 2:
            results = predictor(points=[point], labels=[1])
            center, mask = process_sam_results(results)
        else:
            raise ValueError("No selection made")

    if mask is None:
        print("[WARNING] Could not generate mask")

    return mask


if __name__ == '__main__':
    raise SystemExit("cv_process.py is a library module and is not intended to be run directly.")
