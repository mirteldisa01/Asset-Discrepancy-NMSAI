import cv2

rename_map = {
    "rru": "RRU",
    "panel_antenna": "Panel_Antenna",
    "microwave_dish": "Microwave_Dish"
}

# ==============================
# Process Detection Result
# ==============================
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


# ==============================
# Draw Bounding Boxes on Image
# ==============================
def draw_boxes(image, detections):
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = f'{det["class"]} {det["confidence"]:.2f}'

        # Warna default (BGR)
        color = (0, 255, 0)

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Background label
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(image, (x1, y1 - 25), (x1 + w, y1), color, -1)
        cv2.putText(image, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return image