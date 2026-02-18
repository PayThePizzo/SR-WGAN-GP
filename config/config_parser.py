"""Config parser module.

This module is responsible for:
- Parsing the config/config.yaml file;
- Validating the combination of user-defined parameters;
- Returning a configuration object to define the behavior of the app.
"""

import os
import yaml
from typing import Optional
from pydantic import BaseModel


class LoggingConfig(BaseModel):
    """Configuration for logging during training and validation.

    Attributes:
        interval (int): Frequency (in steps) at which logs are written.
        image_log_count (int): Number of images to log for visualization.
    """
    interval: int = 10
    image_log_count: int = 16
    pass


class DataConfig(BaseModel):
    """Configuration for dataset parameters and loading behavior.

    Attributes:
        dataset (str): Name of the dataset to be used (e.g., "Hand", "40mic").
        img_channels (int): Number of image channels (1 for grayscale, 3 for RGB).
        train_percentage (float): Proportion of data used for training.
        batch_size (int): Batch size for DataLoader.
    """
    dataset: str = "40mic"
    img_channels: int = 3
    train_percentage: float = 0.8
    batch_size: int = 16
    pass


class PreprocessingConfig(BaseModel):
    """Configuration for data augmentation and normalization.

    Attributes:
        normalize (bool): Whether to normalize image tensors.
        normalize_mean (float): Mean value for normalization.
        normalize_std (float): Standard deviation for normalization.
        rotation (bool): Whether to apply random rotation.
        rotation_deg (int): Maximum degrees for random rotation.
        rotation_p (float): Probability of applying rotation.
        flip_horizontal (bool): Whether to apply horizontal flip.
        flip_h_p (float): Probability of horizontal flip.
        flip_vertical (bool): Whether to apply vertical flip.
        flip_v_p (float): Probability of vertical flip.
        gaussian_noise (bool): Whether to apply Gaussian noise.
        noise_percentage (float): Proportion of pixels to alter with noise.
        gaussian_p (float): Probability of applying Gaussian noise.
    """
    normalize: Optional[bool] = True
    normalize_mean: Optional[float] = 0.5
    normalize_std: Optional[float] = 0.5
    rotation: Optional[bool] = True
    rotation_deg: Optional[int] = 10
    rotation_p: Optional[float] = 0.3
    flip_horizontal: Optional[bool] = True
    flip_h_p: Optional[float] = 0.3
    flip_vertical: Optional[bool] = True
    flip_v_p: Optional[float] = 0.3
    gaussian_noise: Optional[bool] = True
    noise_percentage: Optional[float] = 0.1
    gaussian_p: Optional[float] = 0.3
    pass


class TrainingConfig(BaseModel):
    """Configuration for training parameters.

    Attributes:
        epochs (int): Number of training epochs.
    """
    epochs: int = 10
    pass


class ValidatingConfig(BaseModel):
    """Configuration for validation scheduling.

    Attributes:
        interval (int): Validation frequency (in steps).
    """
    interval: int = 10
    pass


class GeneratorConfig(BaseModel):
    """Configuration for generator training behavior.

    Attributes:
        mode (str): Generator architecture mode (e.g., "classic").
        lr (float): Learning rate for the generator.
        beta_1 (float): Beta1 parameter for the Adam optimizer.
        beta_2 (float): Beta2 parameter for the Adam optimizer.
        lambda_adv_loss (float): Weight for adversarial loss.
        lambda_l1_loss (float): Weight for L1 pixel loss.
    """
    mode: str = "classic"
    lr: float = 0.00005
    beta_1: float = 0.5
    beta_2: float = 0.9
    lambda_adv_loss: float = 1.0
    lambda_l1_loss: float = 0.05
    pass


class CriticConfig(BaseModel):
    """Configuration for critic (discriminator) training behavior.

    Attributes:
        mode (str): Critic mode (e.g., "gp" for gradient penalty).
        lr (float): Learning rate for the critic.
        beta_1 (float): Beta1 parameter for the Adam optimizer.
        beta_2 (float): Beta2 parameter for the Adam optimizer.
        lambda_wasserstrein (float): Weight for Wasserstein distance loss.
        lambda_gp (int): Weight for gradient penalty.
        weight_clip (float): Clip value for weights (used if not using gradient penalty).
        c_iter (int): Number of critic updates per generator update.
    """
    mode: str = "gp"
    lr: float = 0.00005
    beta_1: float = 0.5
    beta_2: float = 0.9
    lambda_wasserstrein: float = 0.05
    lambda_gp: int = 10
    weight_clip: float = 0.01
    c_iter: int = 8
    pass


class AppConfig(BaseModel):
    """Top-level application configuration object.

    Groups all sub-configs for logging, data handling, preprocessing,
    training, validating, generator, and critic into one unified model.

    Attributes:
        logging (LoggingConfig): Logging-related configuration.
        data (DataConfig): Data loader and dataset configuration.
        preprocessing (PreprocessingConfig): Image augmentation and normalization.
        training (TrainingConfig): Training schedule configuration.
        validating (ValidatingConfig): Validation schedule configuration.
        generator (GeneratorConfig): Generator-specific training config.
        critic (CriticConfig): Critic-specific training config.
    """
    logging: LoggingConfig
    data: DataConfig
    preprocessing: PreprocessingConfig
    training: TrainingConfig
    validating: ValidatingConfig
    generator: GeneratorConfig
    critic: CriticConfig
    pass


def load_from_config(path: str) -> AppConfig:
    """Load application configuration from a YAML file.

    Parses the specified YAML file and returns an AppConfig object
    composed of validated sub-configs (logging, data, training, etc.).

    Args:
        path (str): Path to the configuration YAML file.

    Returns:
        AppConfig: A fully populated configuration object.
    """
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    return AppConfig(**data)


def save_to_config(
    path: str,
    config: AppConfig,
    name: str = "config",
    str_format: str = "yaml"
) -> None:
    """Save an AppConfig object to disk in YAML or JSON format.

    Args:
        path (str): Destination directory to save the file.
        config (AppConfig): The configuration object to serialize.
        name (str, optional): Filename prefix (without extension). Defaults to "config".
        str_format (str, optional): Output format, either "yaml" or "json". Defaults to "yaml".

    Raises:
        Exception: If an unsupported format is specified.
    """
    dst = os.path.join(path, f"{name}.{str_format}")

    with open(dst, 'w') as f:
        if str_format == "yaml":
            yaml.safe_dump(config.model_dump(), f, indent=2, sort_keys=False)
        elif str_format == "json":
            f.write(config.model_dump_json(indent=2))
        else:
            raise Exception("Accepted formats are yaml or json")

    pass
