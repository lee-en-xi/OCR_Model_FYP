from copy import deepcopy
from glob import glob
import json
import cv2
import os

# output directory
out_dir = "label_studio"
# input images, to match the filename, images should be redownloaded from Label Studio
img_dir = os.path.join("label_studio", "images")
# location of paddleOCR output
all_prelabel_json_fp = glob(os.path.join("output", "*.json"))

all_out_json = []

for start_id, prelabel_json_fp in enumerate(all_prelabel_json_fp):
    with open(prelabel_json_fp, "r") as f:
        in_dict = json.load(f)

    img_filename = os.path.basename(in_dict["input_path"])
    img_path = os.path.join(img_dir, img_filename)

    img = cv2.imread(img_path)
    assert img is not None, f"Failed to read image: {img_path}"

    img_height, img_width = img.shape[:2]

    in_dict = in_dict["overall_ocr_res"]
    polygons = in_dict["rec_polys"]
    texts = in_dict["rec_texts"]

    assert len(polygons) == len(texts)
    result = []
    bb_count = 0
    for p, t in zip(polygons, texts):
        bb_count += 1
        # flatten the list of points to get bounding box
        x_coords = [point[0] for point in p]
        y_coords = [point[1] for point in p]

        min_x = min(x_coords)
        max_x = max(x_coords)
        min_y = min(y_coords)
        max_y = max(y_coords)

        # Normalize to percentage
        x = min_x / img_width * 100
        y = min_y / img_height * 100
        width = (max_x - min_x) / img_width * 100
        height = (max_y - min_y) / img_height * 100

        bbox_info = {
            "original_width": img_width,
            "original_height": img_height,
            "image_rotation": 0,
            "value": {"x": x, "y": y, "width": width, "height": height, "rotation": 0},
            "id": f"bb{bb_count}",
            "from_name": "bbox",
            "to_name": "image",
            "type": "rectangle",
        }

        label_info = deepcopy(bbox_info)
        text_info = deepcopy(bbox_info)

        label_info["value"]["labels"] = ["Text"]
        label_info["from_name"] = "label"
        label_info["type"] = "labels"

        text_info["value"]["text"] = [t]
        text_info["from_name"] = "transcription"
        text_info["type"] = "textarea"

        result.append(bbox_info)
        result.append(label_info)
        result.append(text_info)

    out_dict = {
        "id": start_id,
        "data": {
            "ocr": f"/data/upload/{os.path.basename(img_path)}",
        },
        "predictions": [{"model_version": "v1", "result": result}],
        "score": 0.8,
    }

    all_out_json.append(out_dict)
    start_id += 1

print(f"Converted {len(all_out_json)} images")

with open(os.path.join(out_dir, "prelabeled.json"), "w") as f:
    json.dump(all_out_json, f, indent=2)
