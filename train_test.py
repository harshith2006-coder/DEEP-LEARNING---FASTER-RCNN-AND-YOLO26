import torch
from torch.utils.data import DataLoader

from dataset import (
    CLCXrayDataset,
    IMAGE_DIR,
    ANNOTATION_FILE
)

from model import get_model


# ============================================================
# CONFIGURATION
# ============================================================

BATCH_SIZE = 1

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# COLLATE FUNCTION
# ============================================================
# Detection images can have different sizes, so we cannot use
# the normal PyTorch collate function.

def collate_fn(batch):
    return tuple(zip(*batch))


# ============================================================
# MAIN TEST
# ============================================================

if __name__ == "__main__":

    print("\n======================================")
    print("FASTER R-CNN ONE-BATCH TEST")
    print("======================================")

    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = CLCXrayDataset(
        image_dir=IMAGE_DIR,
        annotation_file=ANNOTATION_FILE,
        train=True
    )

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    data_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )

    print(
        f"Dataset size: {len(dataset)} images"
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("\nLoading model...")

    model = get_model()

    model = model.to(DEVICE)

    # Training mode is required for loss calculation
    model.train()

    print("Model loaded successfully.")

    # --------------------------------------------------------
    # Get ONE batch
    # --------------------------------------------------------

    print("\nLoading one batch...")

    images, targets = next(iter(data_loader))

    print(
        f"Number of images in batch: {len(images)}"
    )

    print(
        f"Image shape: {images[0].shape}"
    )

    print(
        f"Number of boxes: "
        f"{targets[0]['boxes'].shape[0]}"
    )

    # --------------------------------------------------------
    # Move batch to GPU
    # --------------------------------------------------------

    images = [
        image.to(DEVICE)
        for image in images
    ]

    targets = [
        {
            key: value.to(DEVICE)
            for key, value in target.items()
        }
        for target in targets
    ]

    print("Batch moved to device.")

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    print("\nRunning Faster R-CNN forward pass...")

    try:

        loss_dict = model(
            images,
            targets
        )

        print("\nForward pass successful!")

        print("\nLoss components:")

        total_loss = 0.0

        for name, loss in loss_dict.items():

            print(
                f"  {name}: "
                f"{loss.item():.6f}"
            )

            total_loss += loss

        print(
            f"\nTotal loss: "
            f"{total_loss.item():.6f}"
        )

        # ----------------------------------------------------
        # Backward pass
        # ----------------------------------------------------

        print("\nTesting backward pass...")

        total_loss.backward()

        print("Backward pass successful!")

        # ----------------------------------------------------
        # GPU memory
        # ----------------------------------------------------

        if torch.cuda.is_available():

            allocated = (
                torch.cuda.memory_allocated()
                / (1024 ** 3)
            )

            reserved = (
                torch.cuda.memory_reserved()
                / (1024 ** 3)
            )

            print(
                f"\nGPU memory allocated: "
                f"{allocated:.2f} GB"
            )

            print(
                f"GPU memory reserved: "
                f"{reserved:.2f} GB"
            )

        print("\n======================================")
        print("ONE-BATCH TEST PASSED")
        print("======================================")

        print(
            "\nThe dataset, augmentation, model, "
            "forward pass and backward pass are working."
        )

    except RuntimeError as e:

        print("\n======================================")
        print("RUNTIME ERROR")
        print("======================================")

        print(e)

        if "out of memory" in str(e).lower():

            print(
                "\nCUDA OUT OF MEMORY."
            )

            print(
                "We will reduce the memory usage "
                "before training."
            )

        raise