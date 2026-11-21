import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import matplotlib.pyplot as plt
import albumentations as A

from dataset import (
    CLCXrayDataset,
    IMAGE_DIR,
    ANNOTATION_FILE
)

from model import get_model


# ============================================================
# CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# Validation paths
# ------------------------------------------------------------

VAL_IMAGE_DIR = Path(
    r"C:\dl-dataset\val2017"
)

VAL_ANNOTATION_FILE = Path(
    r"C:\dl-dataset\annotations\instances_val2017.json"
)


# ------------------------------------------------------------
# Output directory
# ------------------------------------------------------------

OUTPUT_DIR = Path(
    r"C:\CLCXray_Assignment2\faster_rcnn_results_balanced"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ------------------------------------------------------------
# Training settings
# ------------------------------------------------------------

BATCH_SIZE = 1

# No forced minimum or maximum epochs.
# Training stops automatically using validation loss.
MIN_EPOCHS = 1
MAX_EPOCHS = 30
EARLY_STOP_PATIENCE = 5

MIN_DELTA = 0.0001


# ------------------------------------------------------------
# Optimizer
# ------------------------------------------------------------

LEARNING_RATE = 0.0001

MOMENTUM = 0.9

WEIGHT_DECAY = 0.0005


# ------------------------------------------------------------
# Learning-rate scheduler
# ------------------------------------------------------------

LR_MILESTONES = [20, 27]

LR_GAMMA = 0.1


# ------------------------------------------------------------
# Performance settings
# ------------------------------------------------------------

NUM_WORKERS = 0

PRINT_FREQ = 100

USE_AMP = True

GRAD_CLIP = 5.0


# ------------------------------------------------------------
# Resume this experiment if interrupted.
# This points ONLY to the new balanced-experiment folder.
# ------------------------------------------------------------
RESUME_TRAINING = True

CHECKPOINT_FILE = (
    OUTPUT_DIR / "last_checkpoint.pth"
)


# ============================================================
# RANDOM SEED
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

if DEVICE.type != "cuda":

    raise RuntimeError(
        "CUDA GPU is not available. "
        "This training configuration expects a CUDA GPU."
    )


# ============================================================
# COLLATE FUNCTION
# ============================================================

def collate_fn(batch):

    return tuple(zip(*batch))


# ============================================================
# MINORITY-ONLY AUGMENTATION
# ============================================================

# Faster R-CNN labels are 1-12 because 0 is background.
#
# Minority classes selected for targeted augmentation:
#
# 3  = knife
# 4  = dagger
# 5  = SwissArmyKnife
# 7  = Cans
# 9  = GlassBottle
# 11 = Tin
# 12 = SprayCans

MINORITY_LABELS = {
    3,
    4,
    5,
    7,
    9,
    11,
    12
}

# Controlled image-level oversampling. Majority classes are kept intact.
OVERSAMPLING_FACTORS = {
    3: 1.5,   # knife
    4: 1.5,   # dagger
    5: 1.5,   # SwissArmyKnife
    7: 1.5,   # Cans
    9: 3.0,   # GlassBottle
    11: 1.5,  # Tin
    12: 2.0,  # SprayCans
}


# ------------------------------------------------------------
# Augmentation is applied ONLY to minority-containing images.
# There is NO oversampling.
# ------------------------------------------------------------
minority_augmentation = A.Compose(
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
        )

    ],

    bbox_params=A.BboxParams(
        format="coco",
        label_fields=["labels"],
        clip=True,
        min_visibility=0.3
    )
)


# ============================================================
# MINORITY AUGMENTATION DATASET
# ============================================================

class MinorityAugmentationDataset(Dataset):

    """
    Original dataset is kept unchanged.

    Normal images:
        -> returned unchanged

    Images containing minority classes:
        -> receive targeted augmentation

    IMPORTANT:
        There is NO oversampling.
        Every image is still sampled normally once per epoch.
    """

    def __init__(self, base_dataset):

        self.base_dataset = base_dataset


    def __len__(self):

        return len(self.base_dataset)


    def __getitem__(self, index):

        image, target = self.base_dataset[index]

        labels_tensor = target["labels"]

        labels = labels_tensor.tolist()

        # ----------------------------------------------------
        # Only augment images containing minority classes
        # ----------------------------------------------------

        if not any(
            label in MINORITY_LABELS
            for label in labels
        ):

            return image, target


        # ----------------------------------------------------
        # Convert tensor image to numpy
        # ----------------------------------------------------

        image_np = (
            image.permute(1, 2, 0)
            .cpu()
            .numpy()
        )

        image_np = np.clip(
            image_np * 255.0,
            0,
            255
        ).astype(np.uint8)


        # ----------------------------------------------------
        # Convert XYXY boxes to COCO format
        # ----------------------------------------------------

        boxes_xyxy = (
            target["boxes"]
            .cpu()
            .tolist()
        )

        boxes_coco = []

        for x1, y1, x2, y2 in boxes_xyxy:

            boxes_coco.append(
                [
                    float(x1),
                    float(y1),
                    float(x2 - x1),
                    float(y2 - y1)
                ]
            )


        # ----------------------------------------------------
        # Apply augmentation
        # ----------------------------------------------------

        augmented = minority_augmentation(
            image=image_np,
            bboxes=boxes_coco,
            labels=labels
        )


        image_np = augmented["image"]

        boxes_coco = list(
            augmented["bboxes"]
        )

        labels = list(
            augmented["labels"]
        )


        # ----------------------------------------------------
        # Convert image back to tensor
        # ----------------------------------------------------

        image = (
            torch.from_numpy(
                image_np.copy()
            )
            .permute(2, 0, 1)
            .float()
            / 255.0
        )


        # ----------------------------------------------------
        # Convert COCO boxes back to XYXY
        # ----------------------------------------------------

        boxes_xyxy = []

        for x, y, width, height in boxes_coco:

            boxes_xyxy.append(
                [
                    float(x),
                    float(y),
                    float(x + width),
                    float(y + height)
                ]
            )


        # ----------------------------------------------------
        # Update target
        # ----------------------------------------------------

        if len(boxes_xyxy) > 0:

            target["boxes"] = torch.tensor(
                boxes_xyxy,
                dtype=torch.float32
            )

            target["labels"] = torch.tensor(
                labels,
                dtype=torch.int64
            )

            target["area"] = (
                (
                    target["boxes"][:, 2]
                    - target["boxes"][:, 0]
                )
                *
                (
                    target["boxes"][:, 3]
                    - target["boxes"][:, 1]
                )
            )

        else:

            target["boxes"] = torch.zeros(
                (0, 4),
                dtype=torch.float32
            )

            target["labels"] = torch.zeros(
                (0,),
                dtype=torch.int64
            )

            target["area"] = torch.zeros(
                (0,),
                dtype=torch.float32
            )


        return image, target


# ============================================================
# CONTROLLED MINORITY SAMPLER
# ============================================================

def build_minority_sampler(dataset):
    weights = []

    for index in range(len(dataset)):
        _, target = dataset[index]
        labels = target["labels"].tolist()

        weight = 1.0
        for label in labels:
            weight = max(
                weight,
                OVERSAMPLING_FACTORS.get(int(label), 1.0)
            )

        weights.append(weight)

    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(dataset),
        replacement=True
    )


# ============================================================
# MOVE TARGETS TO DEVICE
# ============================================================

def move_targets_to_device(
    targets,
    device
):

    return [

        {
            key: value.to(device)
            for key, value in target.items()
        }

        for target in targets
    ]


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    data_loader,
    optimizer,
    scaler,
    device,
    epoch
):

    model.train()

    running_loss = 0.0

    start_time = time.time()

    total_batches = len(data_loader)


    for batch_index, (images, targets) in enumerate(
        data_loader
    ):

        images = [

            image.to(
                device,
                non_blocking=True
            )

            for image in images
        ]


        targets = move_targets_to_device(
            targets,
            device
        )


        optimizer.zero_grad(
            set_to_none=True
        )


        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=USE_AMP
        ):

            loss_dict = model(
                images,
                targets
            )

            total_loss = sum(
                loss
                for loss in loss_dict.values()
            )


        loss_value = total_loss.item()


        if not np.isfinite(loss_value):

            raise RuntimeError(
                f"Non-finite loss detected: "
                f"{loss_value}"
            )


        # ----------------------------------------------------
        # Backward pass
        # ----------------------------------------------------

        scaler.scale(
            total_loss
        ).backward()


        # ----------------------------------------------------
        # Gradient clipping
        # ----------------------------------------------------

        scaler.unscale_(
            optimizer
        )

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP
        )


        # ----------------------------------------------------
        # Optimizer update
        # ----------------------------------------------------

        scaler.step(
            optimizer
        )

        scaler.update()


        running_loss += loss_value


        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            (batch_index + 1) % PRINT_FREQ == 0
            or batch_index == 0
            or batch_index + 1 == total_batches
        ):

            elapsed = (
                time.time()
                - start_time
            )

            avg_loss = (
                running_loss
                / (batch_index + 1)
            )

            current_lr = (
                optimizer.param_groups[0]["lr"]
            )

            print(
                f"Epoch [{epoch}] "
                f"Batch [{batch_index + 1}/{total_batches}] "
                f"Loss: {loss_value:.4f} "
                f"Avg: {avg_loss:.4f} "
                f"LR: {current_lr:.7f} "
                f"Time: {elapsed / 60:.1f} min"
            )


    epoch_loss = (
        running_loss
        / len(data_loader)
    )

    return epoch_loss


# ============================================================
# VALIDATION LOSS
# ============================================================

def validate_loss(
    model,
    data_loader,
    device
):

    # Faster R-CNN returns losses in training mode.
    # Gradients are disabled.

    model.train()

    running_loss = 0.0


    with torch.no_grad():

        for images, targets in data_loader:

            images = [

                image.to(
                    device,
                    non_blocking=True
                )

                for image in images
            ]


            targets = move_targets_to_device(
                targets,
                device
            )


            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=USE_AMP
            ):

                loss_dict = model(
                    images,
                    targets
                )

                total_loss = sum(
                    loss
                    for loss in loss_dict.values()
                )


            running_loss += (
                total_loss.item()
            )


    return (
        running_loss
        / len(data_loader)
    )


# ============================================================
# SAVE CHECKPOINT
# ============================================================

def save_checkpoint(
    filename,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    history,
    best_val_loss
):

    checkpoint = {

        "epoch": epoch,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "scheduler_state_dict":
            scheduler.state_dict(),

        "scaler_state_dict":
            scaler.state_dict(),

        "history":
            history,

        "best_val_loss":
            best_val_loss,

        "config": {

            "batch_size":
                BATCH_SIZE,

            "learning_rate":
                LEARNING_RATE,

            "momentum":
                MOMENTUM,

            "weight_decay":
                WEIGHT_DECAY,

            "min_epochs":
                MIN_EPOCHS,

            "max_epochs":
                MAX_EPOCHS,

            "early_stop_patience":
                EARLY_STOP_PATIENCE,

            "amp":
                USE_AMP,

            "minority_augmentation":
                True,

            "oversampling":
                True,

            "oversampling_factors":
                OVERSAMPLING_FACTORS
        }
    }


    torch.save(
        checkpoint,
        OUTPUT_DIR / filename
    )


# ============================================================
# SAVE LOSS GRAPH
# ============================================================

def save_loss_graph(history):

    epochs = range(
        1,
        len(history["train_loss"]) + 1
    )


    plt.figure(
        figsize=(10, 6)
    )


    plt.plot(
        epochs,
        history["train_loss"],
        label="Training Loss"
    )


    plt.plot(
        epochs,
        history["val_loss"],
        label="Validation Loss"
    )


    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Loss"
    )


    plt.title(
        "Faster R-CNN Training and Validation Loss"
    )


    plt.legend()


    plt.grid(
        True,
        alpha=0.3
    )


    plt.tight_layout()


    plt.savefig(
        OUTPUT_DIR / "loss_curve.png",
        dpi=200
    )


    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n=========================================="
    )

    print(
        "FASTER R-CNN TRAINING"
    )

    print(
        "=========================================="
    )


    print(
        f"Device: {DEVICE}"
    )

    print(
        "Model: Faster R-CNN + "
        "MobileNetV3-Large + FPN"
    )

    print(
        f"Batch size: {BATCH_SIZE}"
    )

    print(
        f"Epoch range: {MIN_EPOCHS}-{MAX_EPOCHS}"
    )

    print(
        f"Early-stopping patience: "
        f"{EARLY_STOP_PATIENCE}"
    )

    print(
        f"Learning rate: {LEARNING_RATE}"
    )

    print(
        "Controlled oversampling: ENABLED"
    )
    print("GlassBottle sampling factor: 3.0x")
    print("SprayCans sampling factor: 2.0x")
    print("Other selected minority classes: 1.5x")

    print(
        "Minority-only augmentation: ENABLED"
    )

    print(
        "Augmented classes: "
        "Knife / Dagger / SwissArmyKnife / "
        "Cans / GlassBottle / Tin / SprayCans"
    )

    print(
        f"Momentum: {MOMENTUM}"
    )

    print(
        f"Weight decay: {WEIGHT_DECAY}"
    )

    print(
        f"AMP enabled: {USE_AMP}"
    )


    # ========================================================
    # CHECK VALIDATION PATHS
    # ========================================================

    if not VAL_IMAGE_DIR.exists():

        raise FileNotFoundError(
            f"Validation image folder not found:\n"
            f"{VAL_IMAGE_DIR}"
        )


    if not VAL_ANNOTATION_FILE.exists():

        raise FileNotFoundError(
            f"Validation annotation file not found:\n"
            f"{VAL_ANNOTATION_FILE}"
        )


    # ========================================================
    # TRAIN DATASET
    # ========================================================

    print(
        "\nLoading training dataset..."
    )


    # Original dataset with augmentation disabled.
    base_train_dataset = CLCXrayDataset(
        image_dir=IMAGE_DIR,
        annotation_file=ANNOTATION_FILE,
        train=False
    )


    # Minority-only augmentation wrapper.
    train_dataset = MinorityAugmentationDataset(
        base_train_dataset
    )


    # ========================================================
    # VALIDATION DATASET
    # ========================================================

    print(
        "\nLoading validation dataset..."
    )


    val_dataset = CLCXrayDataset(
        image_dir=VAL_IMAGE_DIR,
        annotation_file=VAL_ANNOTATION_FILE,
        train=False
    )


    print(
        "\n=========================================="
    )


    print(
        f"Original training images: "
        f"{len(base_train_dataset)}"
    )


    print(
        f"Validation images: "
        f"{len(val_dataset)}"
    )


    print(
        "=========================================="
    )


    # ========================================================
    # DATALOADERS
    # ========================================================

    # Controlled minority oversampling.
    train_sampler = build_minority_sampler(
        base_train_dataset
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn
    )


    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn
    )


    # ========================================================
    # MODEL
    # ========================================================

    print(
        "\nLoading Faster R-CNN model..."
    )


    model = get_model()

    model.to(DEVICE)


    print(
        "Model loaded."
    )


    # ========================================================
    # OPTIMIZER
    # ========================================================

    # Differential learning rates:
    # pretrained backbone adapts slowly; detector heads adapt normally.
    backbone_parameters = [
        p for p in model.backbone.parameters()
        if p.requires_grad
    ]

    backbone_ids = {id(p) for p in backbone_parameters}

    head_parameters = [
        p for p in model.parameters()
        if p.requires_grad and id(p) not in backbone_ids
    ]

    optimizer = torch.optim.SGD(
        [
            {
                "params": backbone_parameters,
                "lr": 0.00001
            },
            {
                "params": head_parameters,
                "lr": LEARNING_RATE
            }
        ],
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY
    )


    # ========================================================
    # LR SCHEDULER
    # ========================================================

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=LR_MILESTONES,
        gamma=LR_GAMMA
    )


    # ========================================================
    # AMP SCALER
    # ========================================================

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=USE_AMP
    )


    # ========================================================
    # TRAINING HISTORY
    # ========================================================

    history = {

        "train_loss": [],

        "val_loss": [],

        "learning_rate": []
    }


    best_val_loss = float("inf")

    best_epoch = 0

    epochs_without_improvement = 0
    start_epoch = 1

    if RESUME_TRAINING and CHECKPOINT_FILE.exists():

        print(
            "\n=========================================="
        )
        print("RESUMING BALANCED TRAINING")
        print("==========================================")

        checkpoint = torch.load(
            CHECKPOINT_FILE,
            map_location=DEVICE,
            weights_only=False
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )
        scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )
        scaler.load_state_dict(
            checkpoint["scaler_state_dict"]
        )

        start_epoch = checkpoint["epoch"] + 1
        history = checkpoint.get("history", history)
        best_val_loss = checkpoint.get(
            "best_val_loss",
            float("inf")
        )

        if history["val_loss"]:
            best_epoch = int(
                np.argmin(history["val_loss"])
            ) + 1

        print(f"Resuming from epoch: {start_epoch}")
        print(
            f"Best validation loss so far: "
            f"{best_val_loss:.6f}"
        )

    else:
        print("\nStarting a NEW balanced experiment.")


    # ========================================================
    # TRAINING LOOP
    # ========================================================

    print(
        "\n=========================================="
    )

    print(
        "STARTING TRAINING"
    )

    print(
        "==========================================\n"
    )


    total_training_start = time.time()


    for epoch in range(start_epoch, MAX_EPOCHS + 1):

        epoch_start = time.time()


        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        train_loss = train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=DEVICE,
            epoch=epoch
        )


        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        print(
            "\nCalculating validation loss..."
        )


        val_loss = validate_loss(
            model=model,
            data_loader=val_loader,
            device=DEVICE
        )


        # ----------------------------------------------------
        # CURRENT LR
        # ----------------------------------------------------

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )


        # ----------------------------------------------------
        # SAVE HISTORY
        # ----------------------------------------------------

        history["train_loss"].append(
            train_loss
        )

        history["val_loss"].append(
            val_loss
        )

        history["learning_rate"].append(
            current_lr
        )


        # ----------------------------------------------------
        # EPOCH SUMMARY
        # ----------------------------------------------------

        epoch_time = (
            time.time()
            - epoch_start
        )


        print(
            "\n------------------------------------------"
        )


        print(
            f"Epoch {epoch}"
        )


        print(
            f"Training Loss:   "
            f"{train_loss:.6f}"
        )


        print(
            f"Validation Loss: "
            f"{val_loss:.6f}"
        )


        print(
            f"Learning Rate:   "
            f"{current_lr:.8f}"
        )


        print(
            f"Epoch Time:      "
            f"{epoch_time / 60:.2f} min"
        )


        print(
            "------------------------------------------"
        )


        # ----------------------------------------------------
        # CHECK BEST MODEL
        # ----------------------------------------------------

        improved = (
            val_loss
            < best_val_loss - MIN_DELTA
        )


        if improved:

            best_val_loss = val_loss

            best_epoch = epoch

            epochs_without_improvement = 0


            print(
                "\n*** New best validation loss! ***"
            )


            # Best model weights
            torch.save(
                model.state_dict(),
                OUTPUT_DIR
                / "best_faster_rcnn.pth"
            )


            # Full checkpoint
            save_checkpoint(
                filename="best_checkpoint.pth",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                history=history,
                best_val_loss=best_val_loss
            )


        else:

            epochs_without_improvement += 1


            print(
                f"\nNo improvement. "
                f"Patience: "
                f"{epochs_without_improvement}/"
                f"{EARLY_STOP_PATIENCE}"
            )


        # ----------------------------------------------------
        # LAST CHECKPOINT
        # ----------------------------------------------------

        save_checkpoint(
            filename="last_checkpoint.pth",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            history=history,
            best_val_loss=best_val_loss
        )


        # ----------------------------------------------------
        # UPDATE LR
        # ----------------------------------------------------

        scheduler.step()


        # ----------------------------------------------------
        # SAVE HISTORY + GRAPH
        # ----------------------------------------------------

        with open(
            OUTPUT_DIR / "training_history.json",
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                history,
                f,
                indent=4
            )


        save_loss_graph(
            history
        )


        # ----------------------------------------------------
        # EARLY STOPPING
        # ----------------------------------------------------

        if (
            epoch >= MIN_EPOCHS
            and epochs_without_improvement >= EARLY_STOP_PATIENCE
        ):

            print(
                "\n=========================================="
            )

            print(
                "EARLY STOPPING TRIGGERED"
            )

            print(
                f"No validation improvement for "
                f"{EARLY_STOP_PATIENCE} "
                f"consecutive epochs."
            )

            print(
                f"Best epoch: "
                f"{best_epoch}"
            )

            print(
                f"Best validation loss: "
                f"{best_val_loss:.6f}"
            )

            print(
                "=========================================="
            )

            break


        if epoch == MAX_EPOCHS:
            print(
                "\n=========================================="
            )
            print("MAXIMUM EPOCHS REACHED")
            print(f"Maximum epochs: {MAX_EPOCHS}")
            print("==========================================")
            break

        # ----------------------------------------------------
        # GPU MEMORY
        # ----------------------------------------------------

        allocated = (
            torch.cuda.memory_allocated()
            / (1024 ** 3)
        )

        reserved = (
            torch.cuda.memory_reserved()
            / (1024 ** 3)
        )


        print(
            f"GPU memory: "
            f"{allocated:.2f} GB allocated / "
            f"{reserved:.2f} GB reserved"
        )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    total_training_time = (
        time.time()
        - total_training_start
    )


    print(
        "\n=========================================="
    )

    print(
        "TRAINING COMPLETE"
    )

    print(
        "=========================================="
    )


    print(
        f"Best epoch: "
        f"{best_epoch}"
    )


    print(
        f"Best validation loss: "
        f"{best_val_loss:.6f}"
    )


    print(
        f"Epochs completed: "
        f"{len(history['train_loss'])}"
    )


    print(
        f"Total training time: "
        f"{total_training_time / 3600:.2f} hours"
    )


    print(
        "\nSaved files:"
    )


    print(
        f"  {OUTPUT_DIR / 'best_faster_rcnn.pth'}"
    )

    print(
        f"  {OUTPUT_DIR / 'best_checkpoint.pth'}"
    )

    print(
        f"  {OUTPUT_DIR / 'last_checkpoint.pth'}"
    )

    print(
        f"  {OUTPUT_DIR / 'training_history.json'}"
    )

    print(
        f"  {OUTPUT_DIR / 'loss_curve.png'}"
    )


    print(
        "=========================================="
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()