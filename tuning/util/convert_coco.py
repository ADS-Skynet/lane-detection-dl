import json
import os
import glob

# --- Configuration ---
# Update this to the folder where your 250 .json files are
input_folder = './annotations'
output_filename = '_annotations.coco.json'

# Standard RC Camera Resolution (matches your sample)
WIDTH = 640
HEIGHT = 480

def convert_to_coco():
    coco_output = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "lane"}]
    }

    ann_id = 1
    image_id = 1

    # Find all JSON files in the folder
    json_files = glob.glob(os.path.join(input_folder, "*.json"))

    if not json_files:
        print(f"No JSON files found in {input_folder}!")
        return

    for json_path in json_files:
        with open(json_path, 'r') as f:
            try:
                data = json.load(f)
            except Exception as e:
                print(f"Error reading {json_path}: {e}")
                continue

            # 1. Image Entry
            # Uses the filename defined inside your JSON
            file_name = data.get("image", os.path.basename(json_path).replace('.json', '.jpg'))

            coco_output["images"].append({
                "id": image_id,
                "file_name": file_name,
                "width": WIDTH,
                "height": HEIGHT
            })

            # 2. Annotation Entry (Ribbon Logic)
            for lane in data.get("lanes", []):
                # Skip empty lanes
                if not lane or len(lane) == 0:
                    continue

                # Create a "thick" line by offsetting points
                thickness = 5  # pixels

                # Original points (going up the track)
                side_a = [[p[0], p[1]] for p in lane]

                # Offset points (going back down the track)
                side_b = [[p[0] + thickness, p[1]] for p in lane]
                side_b.reverse()

                # Combine them into one closed loop: A -> B
                full_polygon = side_a + side_b

                # Flatten for COCO: [x1, y1, x2, y2...]
                segmentation = [coord for pt in full_polygon for coord in pt]

                # Calculate Bounding Box
                x_coords = [p[0] for p in full_polygon]
                y_coords = [p[1] for p in full_polygon]
                min_x, max_x = min(x_coords), max(x_coords)
                min_y, max_y = min(y_coords), max(y_coords)

                coco_output["annotations"].append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "segmentation": [segmentation],
                    "area": (max_x - min_x) * (max_y - min_y),
                    "bbox": [min_x, min_y, max_x - min_x, max_y - min_y],
                    "iscrowd": 0
                })
                ann_id += 1

            image_id += 1

    with open(output_filename, 'w') as f:
        json.dump(coco_output, f, indent=4)

    print(f"Done! Created {output_filename} with {image_id-1} images and {ann_id-1} annotations.")

if __name__ == "__main__":
    convert_to_coco()