import json
from collections import Counter

ANNOTATION_FILE = r"C:\dl-dataset\instances_train2017_reduced.json"

# Load COCO annotations
with open(ANNOTATION_FILE, "r", encoding="utf-8") as f:
    coco = json.load(f)

# Map category ID -> class name
category_names = {
    category["id"]: category["name"]
    for category in coco["categories"]
}

# Count object instances per class
class_counts = Counter(
    annotation["category_id"]
    for annotation in coco["annotations"]
)

print("\n======================================")
print("CLASS DISTRIBUTION - REDUCED TRAINING SET")
print("======================================\n")

total_annotations = len(coco["annotations"])

for category_id, name in category_names.items():
    count = class_counts[category_id]
    percentage = (count / total_annotations) * 100

    print(f"{name:<20} {count:>5} objects  ({percentage:>6.2f}%)")

print("\n--------------------------------------")
print(f"Total images      : {len(coco['images'])}")
print(f"Total annotations : {total_annotations}")
print(f"Total classes     : {len(category_names)}")
print("--------------------------------------")