import json
from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset
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

NUM_CLASSES = 12

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

# Faster R-CNN labels:
# 0 = background
# 1 = blade
# 2 = scissors
# 3 = knife
# 4 = dagger
# 5 = SwissArmyKnife
# 6 = PlasticBottle
# 7 = Cans
# 8 = VacuumCup
# 9 = GlassBottle
# 10 = CartonDrinks
# 11 = Tin
# 12 = SprayCans

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
# TRAINING AUGMENTATION
# ============================================================

# Applied ONLY to training images containing
# at least one selected minority class.

train_augmentation = A.Compose(
    [
        # Horizontal flip
        A.HorizontalFlip(
            p=0.5
        ),

        # Mild scale, translation and rotation
        A.Affine(
            scale=(0.95, 1.05),
            translate_percent=(-0.03, 0.03),
            rotate=(-7, 7),
            p=0.7
        ),

        # Mild brightness and contrast
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
# DATASET CLASS
# ============================================================

class CLCXrayDataset(Dataset):

    def __init__(
        self,
        image_dir,
        annotation_file,
        train=False
    ):

        self.image_dir = Path(
            image_dir
        )

        self.train = train

        # ----------------------------------------------------
        # Load COCO annotations
        # ----------------------------------------------------

        with open(
            annotation_file,
            "r",
            encoding="utf-8"
        ) as f:

            self.coco = json.load(f)


        # ----------------------------------------------------
        # Categories
        # ----------------------------------------------------

        self.categories = {
            category["id"]: category["name"]
            for category in self.coco["categories"]
        }


        # ----------------------------------------------------
        # Group annotations by image
        # ----------------------------------------------------

        self.annotations_by_image = {}

        for annotation in self.coco["annotations"]:

            image_id = annotation["image_id"]

            if image_id not in self.annotations_by_image:

                self.annotations_by_image[image_id] = []

            self.annotations_by_image[image_id].append(
                annotation
            )


        # ----------------------------------------------------
        # ORIGINAL IMAGE LIST
        # ----------------------------------------------------

        # IMPORTANT:
        # No oversampling.
        # Every image appears exactly once.

        self.images = self.coco["images"]


        # ----------------------------------------------------
        # Dataset information
        # ----------------------------------------------------

        if self.train:

            minority_image_count = 0
            normal_image_count = 0

            for image_info in self.images:

                image_id = image_info["id"]

                annotations = (
                    self.annotations_by_image.get(
                        image_id,
                        []
                    )
                )

                has_minority = False

                for annotation in annotations:

                    category_id = annotation["category_id"]

                    # Convert COCO category ID
                    # to Faster R-CNN label
                    label = category_id + 1

                    if label in MINORITY_LABELS:

                        has_minority = True
                        break


                if has_minority:

                    minority_image_count += 1

                else:

                    normal_image_count += 1


            print(
                f"Original training images: "
                f"{len(self.images)}"
            )

            print(
                f"Minority-containing images: "
                f"{minority_image_count}"
            )

            print(
                f"Majority-only images: "
                f"{normal_image_count}"
            )

            print(
                "Oversampling: DISABLED"
            )

            print(
                "Minority-only augmentation: ENABLED"
            )

        else:

            print(
                f"Dataset loaded: "
                f"{len(self.images)} images"
            )

            print(
                "Augmentation: DISABLED"
            )

            print(
                "Oversampling: DISABLED"
            )


    # ========================================================
    # LENGTH
    # ========================================================

    def __len__(self):

        return len(self.images)


    # ========================================================
    # GET ITEM
    # ========================================================

    def __getitem__(
        self,
        index
    ):

        image_info = self.images[index]

        image_id = image_info["id"]

        filename = Path(
            image_info["file_name"]
        ).name

        image_path = (
            self.image_dir
            / filename
        )


        # ----------------------------------------------------
        # Load image
        # ----------------------------------------------------

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            raise FileNotFoundError(
                f"Could not load image: "
                f"{image_path}"
            )


        # OpenCV BGR -> RGB

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )


        # ----------------------------------------------------
        # Get annotations
        # ----------------------------------------------------

        annotations = (
            self.annotations_by_image.get(
                image_id,
                []
            )
        )


        boxes = []

        labels = []


        for annotation in annotations:

            x, y, width, height = (
                annotation["bbox"]
            )


            boxes.append(
                [
                    float(x),
                    float(y),
                    float(width),
                    float(height)
                ]
            )


            # ------------------------------------------------
            # COCO IDs:
            # 0–11
            #
            # Faster R-CNN:
            # 0 = background
            # 1–12 = object classes
            # ------------------------------------------------

            labels.append(
                int(
                    annotation["category_id"]
                ) + 1
            )


        # ----------------------------------------------------
        # Determine whether image contains
        # a minority class
        # ----------------------------------------------------

        has_minority_class = False

        if self.train:

            for label in labels:

                if label in MINORITY_LABELS:

                    has_minority_class = True

                    break


        # ----------------------------------------------------
        # MINORITY-ONLY AUGMENTATION
        # ----------------------------------------------------

        if (
            self.train
            and has_minority_class
        ):

            augmented = train_augmentation(
                image=image,
                bboxes=boxes,
                labels=labels
            )

            image = augmented["image"]

            boxes = list(
                augmented["bboxes"]
            )

            labels = list(
                augmented["labels"]
            )


        # ----------------------------------------------------
        # Convert image to tensor
        # ----------------------------------------------------

        image = torch.from_numpy(
            image.copy()
        ).permute(
            2,
            0,
            1
        ).float() / 255.0


        # ----------------------------------------------------
        # Convert COCO boxes:
        #
        # [x, y, width, height]
        #
        # to:
        #
        # [x1, y1, x2, y2]
        # ----------------------------------------------------

        converted_boxes = []

        for box in boxes:

            x, y, width, height = box

            x1 = x
            y1 = y
            x2 = x + width
            y2 = y + height

            converted_boxes.append(
                [
                    x1,
                    y1,
                    x2,
                    y2
                ]
            )


        # ----------------------------------------------------
        # Convert to tensors
        # ----------------------------------------------------

        if len(converted_boxes) == 0:

            boxes_tensor = torch.zeros(
                (0, 4),
                dtype=torch.float32
            )

            labels_tensor = torch.zeros(
                (0,),
                dtype=torch.int64
            )

            areas_tensor = torch.zeros(
                (0,),
                dtype=torch.float32
            )

            iscrowd_tensor = torch.zeros(
                (0,),
                dtype=torch.int64
            )


        else:

            boxes_tensor = torch.tensor(
                converted_boxes,
                dtype=torch.float32
            )

            labels_tensor = torch.tensor(
                labels,
                dtype=torch.int64
            )


            # ------------------------------------------------
            # Bounding-box areas
            # ------------------------------------------------

            areas_tensor = (
                (
                    boxes_tensor[:, 2]
                    - boxes_tensor[:, 0]
                )
                *
                (
                    boxes_tensor[:, 3]
                    - boxes_tensor[:, 1]
                )
            )


            iscrowd_tensor = torch.zeros(
                (
                    len(converted_boxes),
                ),
                dtype=torch.int64
            )


        # ----------------------------------------------------
        # Faster R-CNN target
        # ----------------------------------------------------

        target = {

            "boxes": boxes_tensor,

            "labels": labels_tensor,

            "image_id": torch.tensor(
                [image_id],
                dtype=torch.int64
            ),

            "area": areas_tensor,

            "iscrowd": iscrowd_tensor,

        }


        return image, target


# ============================================================
# TEST DATASET
# ============================================================

if __name__ == "__main__":

    print(
        "\n==================================="
    )

    print(
        "Testing CLCXray Dataset"
    )

    print(
        "===================================\n"
    )


    dataset = CLCXrayDataset(

        image_dir=IMAGE_DIR,

        annotation_file=ANNOTATION_FILE,

        train=True
    )


    print(
        f"\nFinal dataset size: "
        f"{len(dataset)}"
    )


    # --------------------------------------------------------
    # Test several samples
    # --------------------------------------------------------

    print(
        "\nTesting sample loading..."
    )


    test_indices = [
        0,
        min(100, len(dataset) - 1),
        min(500, len(dataset) - 1)
    ]


    for index in test_indices:

        image, target = dataset[index]

        print(
            f"\nSample index: {index}"
        )

        print(
            f"Image tensor shape: "
            f"{image.shape}"
        )

        print(
            f"Bounding boxes: "
            f"{target['boxes'].shape}"
        )

        print(
            f"Labels: "
            f"{target['labels'].tolist()}"
        )


    print(
        "\n==================================="
    )

    print(
        "DATASET TEST COMPLETE"
    )

    print(
        "==================================="
    )