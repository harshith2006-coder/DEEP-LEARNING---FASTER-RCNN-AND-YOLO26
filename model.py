import torch
import torchvision

from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_FPN_Weights
)

from torchvision.models.detection.faster_rcnn import (
    FastRCNNPredictor
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

NUM_CLASSES = 13
# 12 CLCXray classes + 1 background


CLASS_NAMES = [
    "background",
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
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# CREATE MODEL
# ============================================================

def get_model():

    print("Loading Faster R-CNN MobileNetV3-Large FPN...")

    print("Using pretrained COCO weights...")

    # Load Faster R-CNN with MobileNetV3-Large-FPN
    # and COCO pretrained weights
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
        weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
    )

    # --------------------------------------------------------
    # Replace COCO prediction head
    # --------------------------------------------------------

    in_features = (
        model.roi_heads.box_predictor.cls_score.in_features
    )

    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features,
        NUM_CLASSES
    )

    return model


# ============================================================
# TEST MODEL
# ============================================================

if __name__ == "__main__":

    print("\n======================================")
    print("FASTER R-CNN MODEL SETUP")
    print("======================================")

    print(f"Device: {DEVICE}")

    model = get_model()

    model = model.to(DEVICE)

    print("\nModel: Faster R-CNN")
    print("Backbone: MobileNetV3-Large + FPN")
    print("Weights: COCO pretrained")
    print(f"Number of classes: {NUM_CLASSES}")

    for i, name in enumerate(CLASS_NAMES):
        print(f"  {i} = {name}")

    # --------------------------------------------------------
    # Parameter count
    # --------------------------------------------------------

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"\nTotal parameters: "
        f"{total_parameters:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_parameters:,}"
    )

    print("\n======================================")
    print("MODEL SETUP COMPLETE")
    print("======================================")