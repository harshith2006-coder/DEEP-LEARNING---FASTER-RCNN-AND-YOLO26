import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# ==============================
# PATHS
# ==============================

ANNOTATION_FILE = Path(
    r"C:\dl-dataset\instances_train2017_reduced.json"
)

IMAGE_DIR = Path(
    r"C:\dl-dataset\train2017_reduced\train2017_reduced"
)

OUTPUT_DIR = Path(
    r"C:\CLCXray_Assignment2\annotation_check"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==============================
# LOAD COCO ANNOTATIONS
# ==============================

print("Loading annotation file...")

with open(ANNOTATION_FILE, "r", encoding="utf-8") as f:
    coco = json.load(f)

print("Annotation file loaded!")
print(f"Images: {len(coco['images'])}")
print(f"Annotations: {len(coco['annotations'])}")
print(f"Classes: {len(coco['categories'])}")


# ==============================
# CLASS NAMES
# ==============================

category_names = {
    category["id"]: category["name"]
    for category in coco["categories"]
}

print("\nClasses:")
for category_id, name in category_names.items():
    print(f"{category_id}: {name}")


# ==============================
# ORGANIZE ANNOTATIONS
# ==============================

annotations_by_image = {}

for annotation in coco["annotations"]:
    image_id = annotation["image_id"]

    if image_id not in annotations_by_image:
        annotations_by_image[image_id] = []

    annotations_by_image[image_id].append(annotation)


# ==============================
# CHECK FIRST 10 IMAGES
# ==============================

print("\nCreating annotation preview images...")

images_to_check = coco["images"][:10]

for index, image_info in enumerate(images_to_check, start=1):

    image_id = image_info["id"]
    filename = Path(image_info["file_name"]).name

    image_path = IMAGE_DIR / filename

    if not image_path.exists():
        print(f"WARNING: Image not found: {filename}")
        continue

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    image_annotations = annotations_by_image.get(image_id, [])

    print(
        f"{index}. {filename} "
        f"-> {len(image_annotations)} annotations"
    )

    for annotation in image_annotations:

        # COCO format:
        # [x, y, width, height]

        x, y, width, height = annotation["bbox"]

        x1 = int(x)
        y1 = int(y)
        x2 = int(x + width)
        y2 = int(y + height)

        category_id = annotation["category_id"]
        class_name = category_names[category_id]

        # Draw bounding box
        draw.rectangle(
            [x1, y1, x2, y2],
            outline="red",
            width=3
        )

        # Draw class name
        text_position = (x1, max(0, y1 - 20))

        draw.text(
            text_position,
            class_name,
            fill="red"
        )

    # Save result
    output_path = OUTPUT_DIR / f"check_{index:02d}_{filename}"

    image.save(output_path)

print("\n================================")
print("DONE!")
print("================================")
print(f"Preview images saved to:")
print(OUTPUT_DIR)