import cv2

rename_map = {
    "rru": "RRU",
    "panel_antenna": "Panel_Antenna",
    "microwave_dish": "Microwave_Dish"
}

def process_result(result, model):
    detections = []
    object_count = {}

    if result.boxes is None:
        return detections, object_count

    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        class_id = int(box.cls[0])

        original_name = model.names[class_id]
        class_name = rename_map.get(original_name, original_name)

        object_count[class_name] = object_count.get(class_name, 0) + 1

        detections.append({
            "class": class_name,
            "confidence": round(conf, 4),
            "bbox": [x1, y1, x2, y2]
        })

    return detections, object_count