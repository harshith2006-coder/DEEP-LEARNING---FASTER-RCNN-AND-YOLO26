import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from model import get_model
from dataset import CLCXrayDataset

    
# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ------------------------------------------------------------
# TEST DATA
# ------------------------------------------------------------

TEST_IMAGE_DIR = Path(
    r"C:\dl-dataset\val2017"
)

TEST_ANNOTATION_FILE = Path(
    r"C:\dl-dataset\annotations\instances_val2017.json"
)

# ------------------------------------------------------------
# NEW BEST MODEL
# ------------------------------------------------------------

CHECKPOINT_FILE = Path(
    r"C:\CLCXray_Assignment2\faster_rcnn_results_final\best_checkpoint.pth"
)

# ------------------------------------------------------------
# OUTPUT
# ------------------------------------------------------------

OUTPUT_DIR = Path(
    r"C:\CLCXray_Assignment2\faster_rcnn_results_final50"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
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
# COLLATE FUNCTION
# ============================================================

def collate_fn(batch):
    return tuple(zip(*batch))


# ============================================================
# LOAD TEST DATASET
# ============================================================

def create_test_dataset():

    print("\n==========================================")
    print("LOADING TEST DATASET")
    print("==========================================")

    print(f"Test image directory:")
    print(f"  {TEST_IMAGE_DIR}")

    print(f"\nTest annotation file:")
    print(f"  {TEST_ANNOTATION_FILE}")

    if not TEST_IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"Test image directory not found:\n"
            f"{TEST_IMAGE_DIR}"
        )

    if not TEST_ANNOTATION_FILE.exists():
        raise FileNotFoundError(
            f"Test annotation file not found:\n"
            f"{TEST_ANNOTATION_FILE}"
        )

    test_dataset = CLCXrayDataset(
        image_dir=TEST_IMAGE_DIR,
        annotation_file=TEST_ANNOTATION_FILE,
        train=False
    )

    print("\nTest dataset loaded successfully.")
    print(f"Number of test images: {len(test_dataset)}")

    print("\nTest configuration:")
    print("  Augmentation: DISABLED")
    print("  Oversampling: DISABLED")

    return test_dataset


# ============================================================
# LOAD BEST MODEL
# ============================================================

def load_best_model():

    print("\n==========================================")
    print("LOADING BEST MODEL")
    print("==========================================")

    print("Checkpoint:")
    print(f"  {CHECKPOINT_FILE}")

    if not CHECKPOINT_FILE.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n"
            f"{CHECKPOINT_FILE}"
        )

    print("\nCreating Faster R-CNN model...")

    model = get_model()

    print("\nLoading checkpoint...")

    checkpoint = torch.load(
        CHECKPOINT_FILE,
        map_location=DEVICE
    )

    checkpoint_epoch = checkpoint.get(
        "epoch",
        "unknown"
    )

    best_val_loss = checkpoint.get(
        "best_val_loss",
        "unknown"
    )

    print(
        f"Checkpoint epoch: {checkpoint_epoch}"
    )

    print(
        f"Best validation loss: {best_val_loss}"
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(DEVICE)

    model.eval()

    print("\nBest model loaded successfully.")

    return model, checkpoint


# ============================================================
# RUN INFERENCE
# ============================================================

def run_inference(
    model,
    test_loader
):

    print("\n==========================================")
    print("RUNNING TEST INFERENCE")
    print("==========================================")

    predictions = []

    total_images = len(
        test_loader.dataset
    )

    processed = 0

    with torch.no_grad():

        for images, targets in test_loader:

            images = [
                image.to(DEVICE)
                for image in images
            ]

            outputs = model(images)

            for target, output in zip(
                targets,
                outputs
            ):

                image_id = int(
                    target["image_id"].item()
                )

                boxes = (
                    output["boxes"]
                    .detach()
                    .cpu()
                )

                labels = (
                    output["labels"]
                    .detach()
                    .cpu()
                )

                scores = (
                    output["scores"]
                    .detach()
                    .cpu()
                )

                # ------------------------------------------------
                # Keep predictions with score >= 0.05
                # ------------------------------------------------

                keep = scores >= 0.05

                boxes = boxes[keep]
                labels = labels[keep]
                scores = scores[keep]

                for box, label, score in zip(
                    boxes,
                    labels,
                    scores
                ):

                    x1, y1, x2, y2 = (
                        box.tolist()
                    )

                    width = max(
                        0.0,
                        x2 - x1
                    )

                    height = max(
                        0.0,
                        y2 - y1
                    )

                    # Faster R-CNN:
                    # 0 = background
                    # 1-12 = object classes
                    #
                    # COCO:
                    # 0-11 = object classes

                    category_id = (
                        int(label.item()) - 1
                    )

                    if category_id < 0:
                        continue

                    predictions.append(
                        {
                            "image_id": image_id,

                            "category_id": category_id,

                            "bbox": [
                                float(x1),
                                float(y1),
                                float(width),
                                float(height)
                            ],

                            "score": float(
                                score.item()
                            )
                        }
                    )

            processed += 1

            if (
                processed % 50 == 0
                or processed == total_images
            ):

                print(
                    f"Processed "
                    f"{processed}/{total_images} images"
                )

    print("\nInference complete.")

    print(
        f"Total predictions generated: "
        f"{len(predictions)}"
    )

    return predictions


# ============================================================
# COCO EVALUATION
# ============================================================

def evaluate_coco(predictions):

    print("\n==========================================")
    print("COCO OBJECT DETECTION EVALUATION")
    print("==========================================")

    coco_gt = COCO(
        str(TEST_ANNOTATION_FILE)
    )

    if len(predictions) == 0:

        print(
            "\nERROR: No predictions were generated."
        )

        return None

    coco_dt = coco_gt.loadRes(
        predictions
    )

    coco_eval = COCOeval(
        coco_gt,
        coco_dt,
        "bbox"
    )

    # Evaluate all test images
    coco_eval.params.imgIds = (
        coco_gt.getImgIds()
    )

    # Evaluate all categories
    coco_eval.params.catIds = (
        coco_gt.getCatIds()
    )

    print("\nRunning COCO evaluation...")

    coco_eval.evaluate()

    coco_eval.accumulate()

    print("\n==========================================")
    print("OVERALL RESULTS")
    print("==========================================")

    coco_eval.summarize()

    return coco_eval


# ============================================================
# PER-CLASS RESULTS
# ============================================================

def print_per_class_results(
    coco_eval
):

    print("\n==========================================")
    print("PER-CLASS RESULTS")
    print("==========================================")

    precision = coco_eval.eval["precision"]

    cat_ids = coco_eval.params.catIds

    category_names = {
        0: "blade",
        1: "scissors",
        2: "knife",
        3: "dagger",
        4: "SwissArmyKnife",
        5: "PlasticBottle",
        6: "Cans",
        7: "VacuumCup",
        8: "GlassBottle",
        9: "CartonDrinks",
        10: "Tin",
        11: "SprayCans",
    }

    print(
        f"\n{'Class':<20}"
        f"{'AP50':>12}"
        f"{'AP75':>12}"
        f"{'AP':>12}"
    )

    print("-" * 56)

    per_class_results = {}

    for category_index, category_id in enumerate(
        cat_ids
    ):

        class_name = category_names.get(
            category_id,
            f"Category_{category_id}"
        )

        # ----------------------------------------------------
        # AP50
        # ----------------------------------------------------

        precision_50 = precision[
            0,
            :,
            category_index,
            0,
            2
        ]

        precision_50 = precision_50[
            precision_50 > -1
        ]

        if len(precision_50) > 0:

            ap50 = float(
                precision_50.mean()
            )

        else:

            ap50 = 0.0

        # ----------------------------------------------------
        # AP75
        # ----------------------------------------------------

        precision_75 = precision[
            5,
            :,
            category_index,
            0,
            2
        ]

        precision_75 = precision_75[
            precision_75 > -1
        ]

        if len(precision_75) > 0:

            ap75 = float(
                precision_75.mean()
            )

        else:

            ap75 = 0.0

        # ----------------------------------------------------
        # AP 0.50:0.95
        # ----------------------------------------------------

        precision_all = precision[
            :,
            :,
            category_index,
            0,
            2
        ]

        precision_all = precision_all[
            precision_all > -1
        ]

        if len(precision_all) > 0:

            ap = float(
                precision_all.mean()
            )

        else:

            ap = 0.0

        print(
            f"{class_name:<20}"
            f"{ap50:>12.4f}"
            f"{ap75:>12.4f}"
            f"{ap:>12.4f}"
        )

        per_class_results[
            class_name
        ] = {
            "AP50": ap50,
            "AP75": ap75,
            "AP": ap
        }

    return per_class_results


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    coco_eval,
    per_class_results,
    predictions,
    checkpoint
):

    results_file = (
        OUTPUT_DIR /
        "test_evaluation_results.json"
    )

    results = {

        "model": (
            "Faster R-CNN "
            "MobileNetV3-Large + FPN"
        ),

        "checkpoint": str(
            CHECKPOINT_FILE
        ),

        "best_epoch": checkpoint.get(
            "epoch",
            None
        ),

        "best_validation_loss": checkpoint.get(
            "best_val_loss",
            None
        ),

        "test_image_directory": str(
            TEST_IMAGE_DIR
        ),

        "test_annotation_file": str(
            TEST_ANNOTATION_FILE
        ),

        "augmentation": False,

        "oversampling": False,

        "num_test_images": (
            len(coco_eval.params.imgIds)
        ),

        "num_predictions": len(
            predictions
        ),

        "overall": {

            "AP": float(
                coco_eval.stats[0]
            ),

            "AP50": float(
                coco_eval.stats[1]
            ),

            "AP75": float(
                coco_eval.stats[2]
            ),

            "AP_small": float(
                coco_eval.stats[3]
            ),

            "AP_medium": float(
                coco_eval.stats[4]
            ),

            "AP_large": float(
                coco_eval.stats[5]
            ),

            "AR_1": float(
                coco_eval.stats[6]
            ),

            "AR_10": float(
                coco_eval.stats[7]
            ),

            "AR_100": float(
                coco_eval.stats[8]
            ),
        },

        "per_class": (
            per_class_results
        )
    }

    with open(
        results_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=4
        )

    print(
        f"\nResults saved to:\n"
        f"{results_file}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")

    print("=" * 60)

    print(
        "CLCXray FINAL TEST EVALUATION"
    )

    print("=" * 60)

    print(
        f"\nDevice: {DEVICE}"
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    test_dataset = (
        create_test_dataset()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model, checkpoint = (
        load_best_model()
    )

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    predictions = run_inference(
        model,
        test_loader
    )

    # --------------------------------------------------------
    # COCO evaluation
    # --------------------------------------------------------

    coco_eval = evaluate_coco(
        predictions
    )

    if coco_eval is None:
        return

    # --------------------------------------------------------
    # Per-class results
    # --------------------------------------------------------

    per_class_results = (
        print_per_class_results(
            coco_eval
        )
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    save_results(
        coco_eval,
        per_class_results,
        predictions,
        checkpoint
    )

    print("\n")

    print("=" * 60)

    print(
        "TEST EVALUATION COMPLETE"
    )

    print("=" * 60)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()