"""Test module for models."""

import os
import sys
import csv
import lpips
import torch
import numpy as np
from pytorch_fid import fid_score
from torchvision.utils import save_image
from skimage.metrics import structural_similarity as ssim, mean_squared_error
from skimage.metrics import peak_signal_noise_ratio as psnr
from generators import Generator

sys.path.append((os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from main import data_dir
from config.config_parser import load_from_config
from src.data_loader import get_med_mnist_test_loader, get_40mic_test_loader


def super_resolve_test_set(
    model_path: str
):
    """Evaluate a super-resolution model on the test set and report image quality metrics.

    This function loads a trained generator model from a specified folder, applies it to
    a dataset of low-resolution medical images, and computes a variety of perceptual and
    classical image quality metrics (LPIPS, MSE, RMSE, PSNR, SSIM, and FID). It also saves
    the generated and ground truth images, as well as the computed metrics in a CSV file.

    Args:
        model_path (str): Path to the model directory, which must contain `model.pth`
            and `config.yaml`.

    Raises:
        FileNotFoundError: If the specified model path does not exist.
        Exception: If the dataset specified in the config is unsupported.

    Side Effects:
        - Saves output images to the model's output directory.
        - Writes per-image and mean evaluation scores to `benchmark.csv`.
        - Prints evaluation summary to stdout.
    """
    # Set Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Could not find the path to the model: {model_path}")

    # Load config
    config = load_from_config(os.path.join(model_path, 'config.yaml'))
    C = config.data.img_channels
    dataset = config.data.dataset

    # Generated images path
    fake_path = os.path.join(model_path, "output")
    # Test path
    real_path = os.path.join(data_dir, "MedicalMNIST", dataset)

    os.makedirs(os.path.join(fake_path, "generated"), exist_ok=True)
    os.makedirs(os.path.join(fake_path, "original"), exist_ok=True)

    # Load generator
    gen = Generator(img_channels=C)
    checkpoint = torch.load(os.path.join(model_path, 'model.pth'), map_location=device)
    gen.load_state_dict(checkpoint['generator_state_dict'])
    gen.to(device)
    gen.eval()

    # Load test set
    test_loader = None

    if dataset in ["AbdomenCT", "BreastMRI", "CXR", "Hand"]:
        test_loader = get_med_mnist_test_loader(
            img_dir=real_path,
            data_config=config.data
        )
    elif dataset == "40mic":
        test_loader = get_40mic_test_loader(
            img_dir=os.path.join(data_dir, "40mic"),
            data_config=config.data
        )
    else:
        raise Exception("Dataset not found")

    lpips_loss_fn = lpips.LPIPS(net='vgg').to(device)

    scores = []

    for i, (lr, hr) in enumerate(test_loader):
        lr = lr.to(device)
        hr = hr.to(device)

        with torch.no_grad():
            sr = gen(lr)

        # LPIPS needs [-1, 1]
        lpips_val = lpips_loss_fn(sr, hr).mean().item()

        # Normalize range [0,1]
        sr_img = (sr + 1) / 2
        hr_img = (hr + 1) / 2

        # Save images
        save_image(sr_img, os.path.join(fake_path, "generated", f"{i:05d}.png"))
        save_image(hr_img, os.path.join(fake_path, "original", f"{i:05d}.png"))

        # Numpy for classical metrics
        sr_np = sr_img.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        hr_np = hr_img.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        sr_np = np.clip(sr_np, 0, 1)
        hr_np = np.clip(hr_np, 0, 1)

        # Classical metrics
        mse_val = mean_squared_error(hr_np, sr_np)
        rmse_val = np.sqrt(mse_val)
        psnr_val = psnr(hr_np, sr_np, data_range=1.0)
        ssim_val = ssim(hr_np, sr_np, channel_axis=2, data_range=1.0)

        scores.append({
            "id": i,
            "mse": mse_val,
            "rmse": rmse_val,
            "psnr": psnr_val,
            "ssim": ssim_val,
            "lpips": lpips_val,
        })

    # Write csv
    csv_path = os.path.join(model_path, "benchmark.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=scores[0].keys())
        writer.writeheader()
        writer.writerows(scores)

    # Mean metrics
    mean_scores = {
        k: np.mean([s[k] for s in scores])
        for k in ["mse", "rmse", "psnr", "ssim", "lpips"]
    }

    # Compute fid
    fid_score_val = fid_score.calculate_fid_given_paths(
        [
            os.path.join(fake_path, "generated"),
            os.path.join(fake_path, "original")
        ],
        batch_size=32,
        device=device,
        dims=2048
    )

    # Print mean metrics
    print("======== Test Set Evaluation ========")
    for k, v in mean_scores.items():
        print(f"{k.upper()}: {v:.4f}")

    print(f"Fid: {fid_score_val}")
    print(f"Saved metrics to: {csv_path}")

    pass


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: test.py <path_to_trained_model_folder>")
        sys.exit(1)

    super_resolve_test_set(model_path=sys.argv[1])
    pass
