"""
Configuration settings for Antarctic Sea-Ice Forecasting System
"""
import os
from dataclasses import dataclass
from typing import Tuple

@dataclass
class DataConfig:
    """Data-related configuration"""
    data_dir: str = "./data/nsidc_images"  # Path to your downloaded NSIDC images
    image_size: Tuple[int, int] = (256, 256)  # Resize images to this size
    sequence_length: int = 365  # Days of history to use (1 year)
    prediction_horizon: int = 365  # Days to predict ahead
    train_years: Tuple[int, int] = (1979, 2018)  # Training period
    val_years: Tuple[int, int] = (2019, 2021)  # Validation period
    test_years: Tuple[int, int] = (2022, 2024)  # Test period
    

@dataclass
class ModelConfig:
    """Model architecture configuration"""
    # CNN Encoder
    cnn_channels: Tuple[int, ...] = (1, 32, 64, 128, 256)
    cnn_kernel_size: int = 3
    
    # ConvLSTM
    convlstm_hidden_dim: int = 128
    convlstm_kernel_size: int = 3
    convlstm_num_layers: int = 2
    
    # Temporal Transformer
    transformer_dim: int = 256
    transformer_heads: int = 8
    transformer_layers: int = 4
    transformer_dropout: float = 0.1
    
    # Decoder
    decoder_channels: Tuple[int, ...] = (256, 128, 64, 32, 1)
    
    # Uncertainty estimation
    use_uncertainty: bool = True
    mc_dropout_samples: int = 10


@dataclass
class TrainingConfig:
    """Training configuration"""
    batch_size: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    epochs: int = 100
    early_stopping_patience: int = 15
    scheduler_patience: int = 5
    gradient_clip: float = 1.0
    
    # Loss weights for multi-task learning
    ice_loss_weight: float = 1.0
    habitat_loss_weight: float = 0.5
    uncertainty_loss_weight: float = 0.1
    
    # Device
    device: str = "cuda"  # or "cpu"
    num_workers: int = 4
    
    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"


@dataclass
class ForecastConfig:
    """Long-term forecasting configuration"""
    forecast_years: int = 50  # Predict 50 years into future
    autoregressive_steps: int = 365 * 50  # Daily predictions
    ensemble_size: int = 10  # For uncertainty quantification
    

# Create default configs
data_config = DataConfig()
model_config = ModelConfig()
training_config = TrainingConfig()
forecast_config = ForecastConfig()