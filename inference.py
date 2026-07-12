

import argparse
import os

import cv2
import numpy as np
import torch

from model_ViK_FPN_CD import ViK_CD
from checkpoint_utils import load_checkpoint
from seed_utils import seed_everything


def preprocess(img_path: str, size: int = 256) -> torch.Tensor:
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")
    img = cv2.resize(img, (size, size))
    tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    return tensor.unsqueeze(0)  # (1, 3, H, W)


def predict_pair(model: torch.nn.Module, img1_path: str, img2_path: str,
                  device: torch.device, input_size: int = 256) -> np.ndarray:
    x1 = preprocess(img1_path, input_size).to(device)
    x2 = preprocess(img2_path, input_size).to(device)
    with torch.no_grad():
        logits = model(x1, x2)
        pred = torch.softmax(logits, dim=1).argmax(dim=1)  # Table 3: softmax + argmax
    return (pred[0].cpu().numpy() * 255).astype(np.uint8)


def load_model(checkpoint: str, num_classes: int, device: torch.device) -> torch.nn.Module:
    model = ViK_CD(num_classes=num_classes)
    model = load_checkpoint(model, checkpoint, map_location="cpu", strict=False)
    model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--img1", type=str, default=None, help="Path to a single T1 image.")
    parser.add_argument("--img2", type=str, default=None, help="Path to a single T2 image.")
    parser.add_argument("--img1_dir", type=str, default=None, help="Directory of T1 images (batch mode).")
    parser.add_argument("--img2_dir", type=str, default=None, help="Directory of T2 images (batch mode).")
    parser.add_argument("--output", type=str, default="prediction.png", help="Output path (single-pair mode).")
    parser.add_argument("--output_dir", type=str, default="predictions", help="Output directory (batch mode).")
    parser.add_argument("--checkpoint", type=str, required=True,
                         help="Path to a .ckpt file (see checkpoint_utils.load_checkpoint for "
                              "supported formats — both plain and PyTorch-Lightning checkpoints work).")
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--input_size", type=int, default=256)  # Table 3: Input Size
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, args.num_classes, device)

    if args.img1 and args.img2:
        mask = predict_pair(model, args.img1, args.img2, device, args.input_size)
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        cv2.imwrite(args.output, mask)
        print(f"Saved prediction to {args.output}")
    elif args.img1_dir and args.img2_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        filenames = sorted(os.listdir(args.img1_dir))
        for fname in filenames:
            p1 = os.path.join(args.img1_dir, fname)
            p2 = os.path.join(args.img2_dir, fname)
            if not os.path.isfile(p2):
                print(f"Skipping {fname}: no matching file in {args.img2_dir}")
                continue
            mask = predict_pair(model, p1, p2, device, args.input_size)
            cv2.imwrite(os.path.join(args.output_dir, fname), mask)
        print(f"Saved {len(filenames)} predictions to {args.output_dir}")
    else:
        raise ValueError("Provide either --img1/--img2 (single pair) or "
                          "--img1_dir/--img2_dir (batch mode).")


if __name__ == "__main__":
    main()
