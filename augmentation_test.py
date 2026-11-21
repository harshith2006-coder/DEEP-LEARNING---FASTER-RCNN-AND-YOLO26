import json
from pathlib import Path

import cv2
import numpy as np
import albumentations as A


# ============================================
# PATHS
# ============================================

ANNOTATION_FILE = Path(
    r"C:\dl-dataset\instances_train2017_reduced.json"
)

IMAGE_DIR = Path(
    r"C:\dl-dataset\train2017_reduced\train2017_reduced"
)

OUTPUT_DIR = Path(
    r"C:\CLCXray_Assignment2\augmentation_test"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================
# LOAD COCO ANNOTATIONS
# ============================================

print("Loading annotations...")

with open(ANNOTATION_FILE, "r", encoding="utf-8") as f:
    coco = json.load(f)

category_names = {
    category["id"]: category["name"]
    for category in coco["categories"]
}


# ============================================
# SELECT FIRST IMAGE
# ============================================

image_info = coco["images"][0]

image_id = image_info["id"]
filename = Path(image_info["file_name"]).name

image_path = IMAGE_DIR / filename

print(f"Testing image: {filename}")

if not image_path.exists():
    raise FileNotFoundError(
        f"Image not found: {image_path}"
    )


# ============================================
# LOAD IMAGE
# ============================================

image = cv2.imread(str(image_path))

if image is None:
    raise RuntimeError("Could not load image.")

# OpenCV loads BGR → convert to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# ============================================
# GET ANNOTATIONS FOR THIS IMAGE
# ============================================

annotations = [
    ann
    for ann in coco["annotations"]
    if ann["image_id"] == image_id
]

boxes = [
    ann["bbox"]
    for ann in annotations
]

class_labels = [
    category_names[ann["category_id"]]
    for ann in annotations
]

print(f"Number of bounding boxes: {len(boxes)}")

for label, box in zip(class_labels, boxes):
    print(f"  {label}: {box}")


# ============================================
# AUGMENTATION PIPELINE
# ============================================
augmentation = A.Compose(
    [
        # Model input size
        A.Resize(
            height=640,
            width=640
        ),

        # Horizontal flip
        A.HorizontalFlip(
            p=0.5
        ),

        # Mild rotation + translation + scaling
        A.Affine(
            scale=(0.95, 1.05),
            translate_percent=(-0.03, 0.03),
            rotate=(-7, 7),
            p=0.7
        ),

        # Mild brightness / contrast
        A.RandomBrightnessContrast(
            brightness_limit=0.10,
            contrast_limit=0.10,
            p=0.3
        ),
    ],

    bbox_params=A.BboxParams(
        format="coco",
        label_fields=["class_labels"],
        clip=True,
        min_visibility=0.3
    )
)

# ============================================
# DRAW FUNCTION
# ============================================

def draw_boxes(image, boxes, labels):

    image = image.copy()

    for box, label in zip(boxes, labels):

        x, y, w, h = box

        x1 = int(x)
        y1 = int(y)
        x2 = int(x + w)
        y2 = int(y + h)

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            3
        )

        cv2.putText(
            image,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2
        )

    return image


# ============================================
# SAVE ORIGINAL IMAGE
# ============================================

original_visual = draw_boxes(
    image,
    boxes,
    class_labels
)

original_visual = cv2.cvtColor(
    original_visual,
    cv2.COLOR_RGB2BGR
)

original_path = OUTPUT_DIR / "original.png"

cv2.imwrite(
    str(original_path),
    original_visual
)


# ============================================
# APPLY AUGMENTATION
# ============================================

print("\nApplying augmentation...")

result = augmentation(
    image=image,
    bboxes=boxes,
    class_labels=class_labels
)

augmented_image = result["image"]
augmented_boxes = result["bboxes"]
augmented_labels = result["class_labels"]


# ============================================
# PRINT AUGMENTED BOXES
# ============================================

print("\nAugmented bounding boxes:")

for label, box in zip(
    augmented_labels,
    augmented_boxes
):
    print(f"  {label}: {box}")


# ============================================
# SAVE AUGMENTED IMAGE
# ============================================

augmented_visual = draw_boxes(
    augmented_image,
    augmented_boxes,
    augmented_labels
)

augmented_visual = cv2.cvtColor(
    augmented_visual,
    cv2.COLOR_RGB2BGR
)

augmented_path = OUTPUT_DIR / "augmented.png"

cv2.imwrite(
    str(augmented_path),
    augmented_visual
)


# ============================================
# DONE
# ============================================

print("\n======================================")
print("AUGMENTATION TEST COMPLETE")
print("======================================")

print(f"Original image : {original_path}")
print(f"Augmented image: {augmented_path}")

print("\nOriginal boxes :", len(boxes))
print("Augmented boxes:", len(augmented_boxes))