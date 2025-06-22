"""Training modulle for the WGAN."""

import os
import sys
import numpy as np
import torch
import lpips
import torch.optim as optim
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from discriminators import Critic
from generators import Generator
from gradient_penalty import gradient_penalty

sys.path.append((os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from main import data_dir, conf_path
from config.config_parser import load_from_config, save_to_config
from src.med_mnist_data_loader import get_med_mnist_train_val_loaders

# =============================================================================
# Configuration
# =============================================================================
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load config
config = load_from_config(conf_path)

# Data config
dataset = config.data.dataset
C = config.data.img_channels
batch_size = config.data.batch_size

# Training val config
epochs = config.training.epochs

# Generator config
gen_mode = config.generator.mode
lambda_adv = config.generator.lambda_adv_loss
lambda_l1 = config.generator.lambda_l1_loss

# Critic config
critic_mode = config.critic.mode
critic_iter = config.critic.c_iter
lambda_wasserstrein = config.critic.lambda_wasserstrein
use_gp = (critic_mode == "gp")
lambda_gp = config.critic.lambda_gp
weight_clip = config.critic.weight_clip

# Logging config
validating_interval = config.validating.interval
logging_interval = config.logging.interval
image_log_count = config.logging.image_log_count

# =============================================================================
# Output and log dir creation
# =============================================================================
# Set id and dirs
start_time = datetime.now()
model_name = (
    "wgan_" +
    ("gp_" if use_gp else "wc_") +
    f"{dataset}"
)
model_id = f"{model_name}_{start_time.strftime('%Y-%m-%d_%H-%M-%S')}"

# Make output dir and save config
output_dir = f"runs/{model_id}"
os.makedirs(output_dir, exist_ok=True)
save_to_config(output_dir, config)

# Make log dir and tensorboard writers
log_dir = f"logs/{model_id}"
os.makedirs(log_dir, exist_ok=True)
writer_fake = SummaryWriter(f"{log_dir}/fake")
writer_real = SummaryWriter(f"{log_dir}/real")
writer_metrics = SummaryWriter(f"{log_dir}/metrics")

# =============================================================================
# Data loading
# =============================================================================
train_loader = None
val_loader = None

if dataset in ["BreastMRI", "CXR", "Hand"]:
    train_loader, val_loader = get_med_mnist_train_val_loaders(
        img_dir=os.path.join(data_dir, "MedicalMNIST", dataset),
        data_config=config.data
    )
else:
    raise Exception("Dataset not found")

# =============================================================================
# Init models and optimizers
# =============================================================================
gen = Generator(img_channels=C).to(device)
critic = Critic(img_channels=C).to(device)

opt_gen = optim.Adam(
    gen.parameters(), lr=config.generator.lr,
    betas=(config.generator.beta_1, config.generator.beta_2)
)
opt_critic = optim.Adam(
    critic.parameters(), lr=config.critic.lr,
    betas=(config.critic.beta_1, config.critic.beta_2)
)

lpips_loss_fn = lpips.LPIPS(net='vgg').to(device)
# ============================================================================
# Training and validating loop
# ============================================================================
for epoch in range(epochs):
    for batch_idx, (low_res, high_res) in enumerate(train_loader):
        low_res = low_res.to(device)
        high_res = high_res.to(device)
        batch_size = low_res.size(0)

        # Train critic
        for _ in range(critic_iter):
            # Generator fake images
            fake_hr = gen(low_res)

            # Critic realism scores
            critic_real = critic(high_res)
            critic_fake = critic(
                fake_hr.detach()            # Ensure gradients do not backprop into the generator
            )

            gp = 0.0

            # Gradient penalty (enforces Lipschitz constraint)
            if use_gp is True:
                gp = gradient_penalty(critic, high_res, fake_hr, device)

            # Wasserstein distance (negated for minimization objective)
            wasserstrein_dist = -(torch.mean(critic_real) - torch.mean(critic_fake))

            # Critic loss (combination)
            loss_critic = (
                wasserstrein_dist * lambda_wasserstrein +
                gp * lambda_gp
            )

            opt_critic.zero_grad()  # Clear previous gradients
            loss_critic.backward()  # Backpropagate the combined loss
            opt_critic.step()       # Update critic weights

            # Weight clipping (enforces Lipschitz constraint)
            if use_gp is False:
                for p in critic.parameters():
                    p.data.clamp_(-weight_clip, weight_clip)

        # Train generator
        fake_hr = gen(low_res)                  # Upsample low-res input
        critic_fake = critic(fake_hr)           # Critic scores generated images

        # Adversary loss
        loss_adv = -torch.mean(critic_fake)     # Generator wants the critic to think fakes are real

        # Pixel-wise L1 loss
        loss_l1 = torch.nn.functional.l1_loss(fake_hr, high_res)

        # Generator loss (combination)
        loss_gen = (
            loss_adv * lambda_adv +
            loss_l1 * lambda_l1
        )

        opt_gen.zero_grad()     # Clear previous gradients
        loss_gen.backward()     # Backpropagate the combined loss
        opt_gen.step()          # Update generator weights

        # Log metrics to TensorBoard
        global_step = epoch * len(train_loader) + batch_idx
        writer_metrics.add_scalars(
            "Critic vs Generator Loss",
            {
                "Critic Loss": loss_critic.item(),
                "Generator Loss": loss_gen.item()
            },
            global_step
        )
        writer_metrics.add_scalars(
            "Generator/Adjusted L1 Loss vs L1 Loss",
            {
                f"Adjusted L1 - lambda: {lambda_l1}": (loss_l1.item() * lambda_l1),
                "L1": loss_l1.item()
            },
            global_step
        )
        writer_metrics.add_scalars(
            "Generator/Adjusted Adversary Loss vs Adversary Loss",
            {
                f"Adjusted Adv - lambda: {lambda_adv}": (loss_adv.item() * lambda_adv),
                "Adv": loss_adv.item()
            },
            global_step
        )

        if use_gp is True:
            writer_metrics.add_scalars(
                "Critic/Adjusted Gradient Penalty vs Gradient Penalty",
                {
                    f"Adjusted gp - lambda: {lambda_gp}": (gp.item() * lambda_gp),
                    "gp": gp.item()
                },
                global_step
            )

        writer_metrics.add_scalars(
            "Critic/Adjusted Wasserstrein Distance vs Wasserstrein Distance",
            {
                f"Adjusted Wasserstrein - lambda: {lambda_wasserstrein}": (wasserstrein_dist.item() * lambda_wasserstrein),
                "Wasserstrein": wasserstrein_dist.item()
            },
            global_step
        )
        writer_metrics.add_scalars(
            "Critic/Real Output vs Fake Output",
            {
                "Real": torch.mean(critic_real).item(),
                "Fake": torch.mean(critic_fake).item()
            },
            global_step
        )

        # Log images to TensorBoard every log_interval
        if batch_idx % logging_interval == 0:
            with torch.no_grad():
                fake_grid = make_grid(fake_hr[:image_log_count], normalize=True, scale_each=True)
                real_grid = make_grid(high_res[:image_log_count], normalize=True, scale_each=True)
                low_res_grid = make_grid(low_res[:image_log_count], normalize=True, scale_each=True)

                writer_fake.add_image("Images/Generated", fake_grid, global_step)
                writer_real.add_image("Images/Real High-Res", real_grid, global_step)
                writer_real.add_image("Images/Low-Res", low_res_grid, global_step)

            print(
                f"Epoch [{epoch}/{epochs}] Batch {batch_idx}/{len(train_loader)} " +
                f"Loss D: {loss_critic:.4f}, Loss G: {loss_gen:.4f}, L1 Loss: {loss_l1:.4f}"
            )

    # Validation
    if epoch % validating_interval == 0:
        with torch.no_grad():
            mse = 0
            psnr_total = 0
            ssim_total = 0
            lpips_total = 0

            for val_low_res, val_high_res in val_loader:
                val_low_res = val_low_res.to(device)
                val_high_res = val_high_res.to(device)
                val_fake_hr = gen(val_low_res)

                mse += ((val_high_res - val_fake_hr) ** 2).mean().item()

                for i in range(val_fake_hr.size(0)):  # Loop over batch
                    lpips_total += lpips_loss_fn(val_high_res[i], val_fake_hr[i]).mean().item()

                    real = val_high_res[i].cpu().numpy()
                    fake = val_fake_hr[i].cpu().numpy()

                    # Normalize from [-1, 1] to [0, 1]
                    real = (real + 1.0) / 2.0
                    fake = (fake + 1.0) / 2.0

                    real_image = np.transpose(real, (1, 2, 0))
                    fake_image = np.transpose(fake, (1, 2, 0))

                    # Compute total PSNR and SSIM for full image
                    psnr_total += psnr(real_image, fake_image, data_range=1.0)
                    ssim_total += ssim(real_image, fake_image, data_range=1.0, channel_axis=2, win_size=5)

            # Average
            num_samples = len(val_loader.dataset)
            mse /= len(val_loader)
            psnr_avg = psnr_total / num_samples
            ssim_avg = ssim_total / num_samples
            lpips_avg = lpips_total / num_samples

            print(
                "======== Validation =========\n" +
                "-----------------------------\n" +
                f"Samples:  {num_samples} \n" +
                f"MSE:      {mse:.4f} \n" +
                f"RMSE:     {np.sqrt(mse):.4f} \n"
                f"AVG PSNR: {psnr_avg:.4f} \n" +
                f"AVG SSIM: {ssim_avg:.4f} \n" +
                f"AVG LPIPS: {lpips_avg:.4f} \n"
                "-----------------------------"
            )

            # Log validation metrics
            writer_metrics.add_scalar("Validation/Metrics/MSE", mse, global_step=epoch)
            writer_metrics.add_scalar("Validation/Metrics/RMSE", np.sqrt(mse), global_step=epoch)
            writer_metrics.add_scalar("Validation/Metrics/PSNR", psnr_avg, global_step=epoch)
            writer_metrics.add_scalar("Validation/Metrics/SSIM", ssim_avg, global_step=epoch)
            writer_metrics.add_scalar("Validation/Metrics/LPIPS", lpips_avg, global_step=epoch)

            # Log validation images
            val_fake_grid = make_grid(val_fake_hr[:image_log_count], normalize=True, scale_each=True)
            val_real_grid = make_grid(val_high_res[:image_log_count], normalize=True, scale_each=True)
            val_low_res_grid = make_grid(val_low_res[:image_log_count], normalize=True, scale_each=True)

            writer_fake.add_image("Validation/Images/Generated", val_fake_grid, global_step)
            writer_real.add_image("Validation/Images/Real High-Res", val_real_grid, global_step)
            writer_real.add_image("Validation/Images/Low-Res", val_low_res_grid, global_step)

# =============================================================================
# Save models and state
# =============================================================================
model_path = os.path.join(output_dir, 'model.pth')
torch.save({
    'generator_state_dict': gen.state_dict(),
    'critic_state_dict': critic.state_dict(),
    'optimizer_gen_state_dict': opt_gen.state_dict(),
    'optimizer_critic_state_dict': opt_critic.state_dict(),
}, model_path)

writer_fake.close()
writer_real.close()
writer_metrics.close()
