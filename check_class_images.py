import json
from pathlib import Path

import cv2


# ============================================================
# PATHS
# ============================================================

IMAGE_DIR = Path(
    r"C:\dl-dataset\val2017"
)

ANNOTATION_FILE = Path(
    r"C:\dl-dataset\annotations\instances_val2017.json"
)

OUTPUT_DIR = Path(
    r"C:\CLCXray_Assignment2\data_visual_check_val"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CLASSES WE WANT TO INSPECT
# ============================================================

CLASSES_TO_CHECK = [
    "GlassBottle",
    "SprayCans",
    "Cans",
    "knife",
    "dagger",
    "VacuumCup"
]


# Number of images per class
IMAGES_PER_CLASS = 6


# ============================================================
# LOAD COCO
# ============================================================

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


images_by_id = {
    image["id"]: image
    for image in coco["images"]
}


# ============================================================
# GROUP ANNOTATIONS BY CLASS
# ============================================================

class_annotations = {
    class_name: []
    for class_name in CLASSES_TO_CHECK
}


for ann in coco["annotations"]:

    category_id = ann["category_id"]

    class_name = categories.get(
        category_id
    )

    if class_name in class_annotations:

        class_annotations[class_name].append(
            ann
        )


# ============================================================
# DRAW FUNCTION
# ============================================================

def draw_box(
    image,
    bbox,
    label
):

    x, y, w, h = bbox

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
        (x1, max(25, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2
    )


# ============================================================
# PROCESS EACH CLASS
# ============================================================

print(
    "\n=========================================="
)

print(
    "CREATING CLASS VISUALIZATIONS"
)

print(
    "=========================================="
)


for class_name in CLASSES_TO_CHECK:

    print(
        f"\nProcessing: {class_name}"
    )


    annotations = class_annotations[
        class_name
    ]


    # --------------------------------------------------------
    # Select different images
    # --------------------------------------------------------

    selected = []

    seen_images = set()


    for ann in annotations:

        image_id = ann["image_id"]

        if image_id in seen_images:

            continue

        seen_images.add(
            image_id
        )

        selected.append(
            image_id
        )

        if len(selected) >= IMAGES_PER_CLASS:

            break


    # --------------------------------------------------------
    # Save individual images
    # --------------------------------------------------------

    class_dir = (
        OUTPUT_DIR / class_name
    )

    class_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    for number, image_id in enumerate(
        selected,
        start=1
    ):

        image_info = images_by_id[
            image_id
        ]

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
                f"Could not load: {filename}"
            )

            continue


        # ----------------------------------------------------
        # Draw ALL annotations in the image
        # ----------------------------------------------------

        all_annotations = [
            ann
            for ann in coco["annotations"]
            if ann["image_id"] == image_id
        ]


        for ann in all_annotations:

            category_name = categories.get(
                ann["category_id"],
                "Unknown"
            )

            draw_box(
                image,
                ann["bbox"],
                category_name
            )


        output_path = (
            class_dir
            / f"{number}_{filename}"
        )


        cv2.imwrite(
            str(output_path),
            image
        )


        print(
            f"  Saved: {output_path}"
        )


# ============================================================
# DONE
# ============================================================

print(
    "\n=========================================="
)

print(
    "VISUAL CHECK COMPLETE"
)

print(
    "=========================================="
)

print(
    f"\nImages saved to:"
)

print(
    OUTPUT_DIR
)

print(
    "\nOpen the folders for:"
)

for class_name in CLASSES_TO_CHECK:

    print(
        f"  - {class_name}"
    )