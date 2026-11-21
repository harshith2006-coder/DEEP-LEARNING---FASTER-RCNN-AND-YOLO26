import json
from pathlib import Path

import cv2
import albumentations as A


# ============================================================
# PATHS
# ============================================================

IMAGE_DIR = Path(
    r"C:\dl-dataset\train2017_reduced\train2017_reduced"
)

ANNOTATION_FILE = Path(
    r"C:\dl-dataset\instances_train2017_reduced.json"
)


# ============================================================
# CLASS INFORMATION
# ============================================================

CLASS_NAMES = [
    "blade",
    "scissors",
    "knife",
    "dagger",
    "SwissArmyKnife",
    "PlasticBottle",
    "Cans",
    "VacuumCup",
    "GlassBottle",
    "CartonDrinks",
    "Tin",
    "SprayCans",
]


# ============================================================
# MINORITY CLASSES
# ============================================================

MINORITY_LABELS = {
    3,      # knife
    4,      # dagger
    5,      # SwissArmyKnife
    7,      # Cans
    9,      # GlassBottle
    11,     # Tin
    12      # SprayCans
}


# ============================================================
# SAME AUGMENTATION USED FOR TRAINING
# ============================================================

augmentation = A.Compose(
    [
        A.HorizontalFlip(
            p=0.5
        ),

        A.Affine(
            scale=(0.95, 1.05),
            translate_percent=(-0.03, 0.03),
            rotate=(-7, 7),
            p=0.7
        ),

        A.RandomBrightnessContrast(
            brightness_limit=0.10,
            contrast_limit=0.10,
            p=0.3
        ),
    ],

    bbox_params=A.BboxParams(
        format="coco",
        label_fields=["labels"],
        clip=True,
        min_visibility=0.3
    )
)


# ============================================================
# LOAD DATASET
# ============================================================

print("\n==========================================")
print("DATASET DIAGNOSTIC")
print("==========================================\n")

print("Loading annotations...")

with open(
    ANNOTATION_FILE,
    "r",
    encoding="utf-8"
) as f:

    coco = json.load(f)


categories = {
    category["id"]: category["name"]
    for category in coco["categories"]
}


images = coco["images"]
annotations = coco["annotations"]


print(
    f"Total images: {len(images)}"
)

print(
    f"Total annotations: {len(annotations)}"
)


# ============================================================
# GROUP ANNOTATIONS BY IMAGE
# ============================================================

annotations_by_image = {}

for ann in annotations:

    image_id = ann["image_id"]

    if image_id not in annotations_by_image:

        annotations_by_image[image_id] = []

    annotations_by_image[image_id].append(
        ann
    )


# ============================================================
# CLASS COUNTS
# ============================================================

object_counts = {
    name: 0
    for name in CLASS_NAMES
}

image_counts = {
    name: 0
    for name in CLASS_NAMES
}


# ============================================================
# CHECK ORIGINAL BOXES
# ============================================================

invalid_original_boxes = []

images_with_invalid_boxes = set()


for image_info in images:

    image_id = image_info["id"]

    image_annotations = (
        annotations_by_image.get(
            image_id,
            []
        )
    )

    classes_in_image = set()


    for ann in image_annotations:

        category_id = ann["category_id"]

        class_name = categories.get(
            category_id,
            f"UNKNOWN_{category_id}"
        )

        if class_name in object_counts:

            object_counts[class_name] += 1

            classes_in_image.add(
                class_name
            )


        x, y, width, height = (
            ann["bbox"]
        )


        # ----------------------------------------------------
        # Check original COCO box
        # ----------------------------------------------------

        if (
            width <= 0
            or height <= 0
        ):

            invalid_original_boxes.append(
                {
                    "image_id": image_id,
                    "file_name": image_info["file_name"],
                    "class": class_name,
                    "bbox": ann["bbox"]
                }
            )

            images_with_invalid_boxes.add(
                image_id
            )


    for class_name in classes_in_image:

        if class_name in image_counts:

            image_counts[class_name] += 1


# ============================================================
# PRINT CLASS DISTRIBUTION
# ============================================================

print(
    "\n=========================================="
)

print(
    "CLASS DISTRIBUTION"
)

print(
    "=========================================="
)

print(
    f"{'Class':<20}"
    f"{'Objects':>10}"
    f"{'Images':>10}"
)

print(
    "-" * 40
)


for class_name in CLASS_NAMES:

    print(
        f"{class_name:<20}"
        f"{object_counts[class_name]:>10}"
        f"{image_counts[class_name]:>10}"
    )


# ============================================================
# ORIGINAL BOX RESULTS
# ============================================================

print(
    "\n=========================================="
)

print(
    "ORIGINAL BOUNDING BOX CHECK"
)

print(
    "=========================================="
)


print(
    f"Invalid original boxes: "
    f"{len(invalid_original_boxes)}"
)

print(
    f"Images containing invalid boxes: "
    f"{len(images_with_invalid_boxes)}"
)


if invalid_original_boxes:

    print(
        "\nFirst invalid boxes:"
    )

    for item in invalid_original_boxes[:20]:

        print(
            f"  Image: {item['file_name']}"
        )

        print(
            f"  Class: {item['class']}"
        )

        print(
            f"  Box: {item['bbox']}"
        )

        print()


# ============================================================
# CHECK AUGMENTED BOXES
# ============================================================

print(
    "\n=========================================="
)

print(
    "AUGMENTATION BOUNDING BOX CHECK"
)

print(
    "=========================================="
)


invalid_augmented_boxes = []

augmented_images_checked = 0

augmentation_failures = 0


for image_index, image_info in enumerate(images):

    image_id = image_info["id"]

    image_annotations = (
        annotations_by_image.get(
            image_id,
            []
        )
    )


    if not image_annotations:

        continue


    # --------------------------------------------------------
    # Determine labels
    # --------------------------------------------------------

    labels = []

    boxes = []

    has_minority = False


    for ann in image_annotations:

        category_id = ann["category_id"]

        label = category_id + 1

        labels.append(label)

        boxes.append(
            [
                float(ann["bbox"][0]),
                float(ann["bbox"][1]),
                float(ann["bbox"][2]),
                float(ann["bbox"][3])
            ]
        )


        if label in MINORITY_LABELS:

            has_minority = True


    # --------------------------------------------------------
    # Only images that actually receive augmentation
    # --------------------------------------------------------

    if not has_minority:

        continue


    filename = Path(
        image_info["file_name"]
    ).name

    image_path = (
        IMAGE_DIR / filename
    )


    image = cv2.imread(
        str(image_path)
    )


    if image is None:

        print(
            f"WARNING: Could not load "
            f"{image_path}"
        )

        continue


    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )


    augmented_images_checked += 1


    try:

        result = augmentation(
            image=image,
            bboxes=boxes,
            labels=labels
        )


    except Exception as e:

        augmentation_failures += 1

        print(
            "\nAUGMENTATION ERROR"
        )

        print(
            f"Image: {filename}"
        )

        print(
            f"Error: {type(e).__name__}: {e}"
        )

        continue


    augmented_boxes = result[
        "bboxes"
    ]

    augmented_labels = result[
        "labels"
    ]


    # --------------------------------------------------------
    # Check every augmented box
    # --------------------------------------------------------

    for box, label in zip(
        augmented_boxes,
        augmented_labels
    ):

        x, y, width, height = box

        x1 = x

        y1 = y

        x2 = x + width

        y2 = y + height


        if (
            x2 <= x1
            or y2 <= y1
            or width <= 0
            or height <= 0
        ):

            class_index = (
                int(label) - 1
            )

            if (
                0 <= class_index
                < len(CLASS_NAMES)
            ):

                class_name = (
                    CLASS_NAMES[class_index]
                )

            else:

                class_name = (
                    f"UNKNOWN_LABEL_{label}"
                )


            invalid_augmented_boxes.append(
                {
                    "image": filename,
                    "class": class_name,
                    "box": box
                }
            )


# ============================================================
# AUGMENTATION RESULTS
# ============================================================

print(
    f"Augmented images checked: "
    f"{augmented_images_checked}"
)

print(
    f"Augmentation failures: "
    f"{augmentation_failures}"
)

print(
    f"Invalid augmented boxes: "
    f"{len(invalid_augmented_boxes)}"
)


if invalid_augmented_boxes:

    print(
        "\nFirst invalid augmented boxes:"
    )

    for item in invalid_augmented_boxes[:20]:

        print(
            f"  Image: {item['image']}"
        )

        print(
            f"  Class: {item['class']}"
        )

        print(
            f"  Box: {item['box']}"
        )

        print()


# ============================================================
# FINAL SUMMARY
# ============================================================

print(
    "\n=========================================="
)

print(
    "FINAL DIAGNOSTIC SUMMARY"
)

print(
    "=========================================="
)


if len(invalid_original_boxes) == 0:

    print(
        "✓ Original annotations: CLEAN"
    )

else:

    print(
        "✗ Original annotations: "
        f"{len(invalid_original_boxes)} "
        "invalid boxes found"
    )


if len(invalid_augmented_boxes) == 0:

    print(
        "✓ Augmented boxes: CLEAN"
    )

else:

    print(
        "✗ Augmented boxes: "
        f"{len(invalid_augmented_boxes)} "
        "invalid boxes found"
    )


if augmentation_failures == 0:

    print(
        "✓ Augmentation: No failures"
    )

else:

    print(
        "✗ Augmentation failures: "
        f"{augmentation_failures}"
    )


print(
    "\n=========================================="
)

print(
    "DIAGNOSTIC COMPLETE"
)

print(
    "=========================================="
)