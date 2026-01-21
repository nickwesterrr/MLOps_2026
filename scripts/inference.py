import sys

import torch

from ml_core.data import get_dataloaders
from ml_core.models import MLP
from ml_core.utils import load_config


def run_inference(checkpoint_path, config_path):
    # 1. Load Config and Device
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Re-initialize Model Structure
    model = MLP(
        input_shape=config["data"]["input_shape"],
        hidden_units=config["model"]["hidden_units"],
        dropout_rate=config["model"]["dropout_rate"],
        num_classes=config["model"]["num_classes"],
    ).to(device)

    # 3. Load Checkpoint
    # weights_only=False is required because we saved the config dict (with numpy types) in the pt file
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 4. Run on a few validation samples
    # We use the dataloader from the config to get real data
    _, val_loader = get_dataloaders(config)

    # Get one batch
    images, labels = next(iter(val_loader))
    images = images.to(device)

    with torch.no_grad():
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

    print("\n Inference Results ")
    print(f"True Labels:        {labels[:5].numpy()}")
    print(f"Model Predictions:  {preds[:5].cpu().numpy()}")
    print(f"Tumor Probability:  {probs[:5, 1].cpu().numpy()}")

    # Simple check
    match = (labels[:5] == preds[:5].cpu()).sum().item()
    print(f"Matches in first 5: {match}/5")
    print("Success!")


if __name__ == "__main__":
    # Checks if arguments are provided
    if len(sys.argv) < 3:
        print("Usage: python scripts/inference.py <checkpoint_path> <config_path>")
        sys.exit(1)

    checkpoint_file = sys.argv[1]
    config_file = sys.argv[2]

    # Run the function
    run_inference(checkpoint_file, config_file)
