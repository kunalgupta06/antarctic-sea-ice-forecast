# """
# Training script for Antarctic Sea-Ice Forecasting Model
# Includes multi-task loss, uncertainty-aware training, and comprehensive logging
# """
# import os
# import time
# import json
# from datetime import datetime
# from typing import Dict, Optional, Tuple
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.optim import AdamW
# from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
# from torch.utils.tensorboard import SummaryWriter
# from tqdm import tqdm

# from config import TrainingConfig, DataConfig, ModelConfig
# from model import AntarcticSeaIceForecaster, create_model
# from data_loader import create_dataloaders


# class SeaIceLoss(nn.Module):
#     """
#     Multi-task loss for sea-ice forecasting
#     Combines reconstruction loss, structural similarity, and edge-aware losses
#     """
    
#     def __init__(self, use_uncertainty: bool = True):
#         super().__init__()
#         self.use_uncertainty = use_uncertainty
#         self.mse = nn.MSELoss(reduction='none')
#         self.l1 = nn.L1Loss(reduction='none')
        
#     def edge_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#         """Loss on ice edges (melt boundaries)"""
#         # Sobel filters for edge detection
#         sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=pred.dtype, device=pred.device)
#         sobel_y = sobel_x.T
        
#         sobel_x = sobel_x.view(1, 1, 3, 3)
#         sobel_y = sobel_y.view(1, 1, 3, 3)
        
#         # Edge maps
#         pred_edges_x = F.conv2d(pred, sobel_x, padding=1)
#         pred_edges_y = F.conv2d(pred, sobel_y, padding=1)
#         pred_edges = torch.sqrt(pred_edges_x**2 + pred_edges_y**2 + 1e-6)
        
#         target_edges_x = F.conv2d(target, sobel_x, padding=1)
#         target_edges_y = F.conv2d(target, sobel_y, padding=1)
#         target_edges = torch.sqrt(target_edges_x**2 + target_edges_y**2 + 1e-6)
        
#         return F.mse_loss(pred_edges, target_edges)
    
#     def ssim_loss(self, pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> torch.Tensor:
#         """Structural Similarity Index Loss"""
#         C1 = 0.01 ** 2
#         C2 = 0.03 ** 2
        
#         # Gaussian window
#         sigma = 1.5
#         coords = torch.arange(window_size, dtype=pred.dtype, device=pred.device) - window_size // 2
#         g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
#         window = g.unsqueeze(0) * g.unsqueeze(1)
#         window = window / window.sum()
#         window = window.view(1, 1, window_size, window_size)
        
#         mu_pred = F.conv2d(pred, window, padding=window_size // 2)
#         mu_target = F.conv2d(target, window, padding=window_size // 2)
        
#         mu_pred_sq = mu_pred ** 2
#         mu_target_sq = mu_target ** 2
#         mu_pred_target = mu_pred * mu_target
        
#         sigma_pred_sq = F.conv2d(pred ** 2, window, padding=window_size // 2) - mu_pred_sq
#         sigma_target_sq = F.conv2d(target ** 2, window, padding=window_size // 2) - mu_target_sq
#         sigma_pred_target = F.conv2d(pred * target, window, padding=window_size // 2) - mu_pred_target
        
#         ssim = ((2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)) / \
#                ((mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2))
        
#         return 1 - ssim.mean()
    
#     def forward(
#         self,
#         predictions: Dict[str, torch.Tensor],
#         targets: torch.Tensor
#     ) -> Tuple[torch.Tensor, Dict[str, float]]:
#         """
#         Args:
#             predictions: dict with 'ice_maps', 'uncertainty', etc.
#             targets: (B, T, 1, H, W) ground truth
#         Returns:
#             total_loss, loss_dict
#         """
#         pred_ice = predictions['ice_maps']
#         B, T, C, H, W = pred_ice.shape
        
#         # Ensure targets match prediction shape
#         if targets.shape[1] > T:
#             targets = targets[:, :T]
#         elif targets.shape[1] < T:
#             pred_ice = pred_ice[:, :targets.shape[1]]
#             T = targets.shape[1]
        
#         # Resize if needed
#         if targets.shape[-2:] != pred_ice.shape[-2:]:
#             targets = F.interpolate(
#                 targets.view(B * T, C, -1, targets.shape[-1]),
#                 size=pred_ice.shape[-2:],
#                 mode='bilinear',
#                 align_corners=False
#             ).view(B, T, C, *pred_ice.shape[-2:])
        
#         losses = {}
        
#         # Reconstruction loss (MSE)
#         mse_loss = F.mse_loss(pred_ice, targets)
#         losses['mse'] = mse_loss.item()
        
#         # L1 loss (more robust to outliers)
#         l1_loss = F.l1_loss(pred_ice, targets)
#         losses['l1'] = l1_loss.item()
        
#         # Edge-aware loss
#         pred_flat = pred_ice.view(-1, 1, H, W)
#         target_flat = targets.view(-1, 1, H, W)
#         edge_loss = self.edge_loss(pred_flat, target_flat)
#         losses['edge'] = edge_loss.item()
        
#         # SSIM loss
#         ssim_loss = self.ssim_loss(pred_flat, target_flat)
#         losses['ssim'] = ssim_loss.item()
        
#         # Total reconstruction loss
#         recon_loss = 0.5 * mse_loss + 0.3 * l1_loss + 0.1 * edge_loss + 0.1 * ssim_loss
        
#         # Uncertainty-aware loss (Gaussian NLL)
#         if self.use_uncertainty and 'uncertainty' in predictions:
#             uncertainty = predictions['uncertainty']
#             variance = uncertainty ** 2 + 1e-6
            
#             nll_loss = 0.5 * (torch.log(variance) + (pred_ice - targets) ** 2 / variance)
#             nll_loss = nll_loss.mean()
#             losses['nll'] = nll_loss.item()
            
#             total_loss = recon_loss + 0.1 * nll_loss
#         else:
#             total_loss = recon_loss
        
#         losses['total'] = total_loss.item()
        
#         return total_loss, losses


# class HabitatRiskLoss(nn.Module):
#     """Loss for habitat risk prediction"""
    
#     def __init__(self):
#         super().__init__()
#         self.ce = nn.CrossEntropyLoss()
        
#     def forward(
#         self,
#         pred_class: torch.Tensor,
#         pred_map: torch.Tensor,
#         ice_concentration: torch.Tensor
#     ) -> torch.Tensor:
#         """
#         Compute habitat risk loss based on ice concentration
#         Lower ice = higher risk
#         """
#         # Derive pseudo-labels from ice concentration
#         # Mean ice concentration per sample
#         mean_ice = ice_concentration.mean(dim=(-2, -1))  # (B, T, 1)
        
#         # Bin into 5 risk categories
#         risk_labels = torch.zeros_like(mean_ice, dtype=torch.long)
#         risk_labels[mean_ice < 0.2] = 4  # Critical risk
#         risk_labels[(mean_ice >= 0.2) & (mean_ice < 0.4)] = 3  # High risk
#         risk_labels[(mean_ice >= 0.4) & (mean_ice < 0.6)] = 2  # Moderate risk
#         risk_labels[(mean_ice >= 0.6) & (mean_ice < 0.8)] = 1  # Low risk
#         risk_labels[mean_ice >= 0.8] = 0  # Minimal risk
        
#         risk_labels = risk_labels.squeeze(-1)  # (B, T)
        
#         # Classification loss
#         B, T, num_classes = pred_class.shape
#         pred_class_flat = pred_class.view(-1, num_classes)
#         risk_labels_flat = risk_labels.view(-1)
        
#         class_loss = self.ce(pred_class_flat, risk_labels_flat)
        
#         # Spatial consistency: risk map should be high where ice is low
#         expected_risk_map = 1 - ice_concentration
#         map_loss = F.mse_loss(pred_map, expected_risk_map)
        
#         return class_loss + 0.5 * map_loss


# class Trainer:
#     """Main training class"""
    
#     def __init__(
#         self,
#         model: AntarcticSeaIceForecaster,
#         train_loader,
#         val_loader,
#         config: TrainingConfig,
#         device: str = 'cuda'
#     ):
#         self.model = model.to(device)
#         self.train_loader = train_loader
#         self.val_loader = val_loader
#         self.config = config
#         self.device = device
        
#         # Losses
#         self.ice_loss = SeaIceLoss(use_uncertainty=model.use_uncertainty)
#         self.habitat_loss = HabitatRiskLoss()
        
#         # Optimizer
#         self.optimizer = AdamW(
#             model.parameters(),
#             lr=config.learning_rate,
#             weight_decay=config.weight_decay
#         )
        
#         # Scheduler
#         self.scheduler = ReduceLROnPlateau(
#             self.optimizer,
#             mode='min',
#             factor=0.5,
#             patience=config.scheduler_patience,
#             verbose=True
#         )
        
#         # Logging
#         os.makedirs(config.log_dir, exist_ok=True)
#         os.makedirs(config.checkpoint_dir, exist_ok=True)
#         self.writer = SummaryWriter(config.log_dir)
        
#         # State
#         self.epoch = 0
#         self.best_val_loss = float('inf')
#         self.patience_counter = 0
        
#     def train_epoch(self) -> Dict[str, float]:
#         """Train for one epoch"""
#         self.model.train()
#         epoch_losses = {'total': 0, 'ice': 0, 'habitat': 0}
        
#         pbar = tqdm(self.train_loader, desc=f'Epoch {self.epoch}')
        
#         for batch_idx, (inputs, targets, metadata) in enumerate(pbar):
#             inputs = inputs.to(self.device)
#             targets = targets.to(self.device)
            
#             # Forward pass
#             self.optimizer.zero_grad()
#             predictions = self.model(inputs, forecast_steps=targets.shape[1])
            
#             # Compute losses
#             ice_loss, ice_losses = self.ice_loss(predictions, targets)
            
#             habitat_loss = self.habitat_loss(
#                 predictions['habitat_risk_class'],
#                 predictions['habitat_risk_map'],
#                 predictions['ice_maps']
#             )
            
#             # Total loss
#             total_loss = (
#                 self.config.ice_loss_weight * ice_loss +
#                 self.config.habitat_loss_weight * habitat_loss
#             )
            
#             # Backward pass
#             total_loss.backward()
            
#             # Gradient clipping
#             torch.nn.utils.clip_grad_norm_(
#                 self.model.parameters(),
#                 self.config.gradient_clip
#             )
            
#             self.optimizer.step()
            
#             # Update metrics
#             epoch_losses['total'] += total_loss.item()
#             epoch_losses['ice'] += ice_loss.item()
#             epoch_losses['habitat'] += habitat_loss.item()
            
#             pbar.set_postfix({
#                 'loss': f"{total_loss.item():.4f}",
#                 'ice': f"{ice_loss.item():.4f}",
#             })
        
#         # Average losses
#         num_batches = len(self.train_loader)
#         for k in epoch_losses:
#             epoch_losses[k] /= num_batches
            
#         return epoch_losses
    
#     @torch.no_grad()
#     def validate(self) -> Dict[str, float]:
#         """Validate the model"""
#         self.model.eval()
#         val_losses = {'total': 0, 'ice': 0, 'habitat': 0}
        
#         for inputs, targets, metadata in tqdm(self.val_loader, desc='Validation'):
#             inputs = inputs.to(self.device)
#             targets = targets.to(self.device)
            
#             predictions = self.model(inputs, forecast_steps=targets.shape[1])
            
#             ice_loss, _ = self.ice_loss(predictions, targets)
#             habitat_loss = self.habitat_loss(
#                 predictions['habitat_risk_class'],
#                 predictions['habitat_risk_map'],
#                 predictions['ice_maps']
#             )
            
#             total_loss = (
#                 self.config.ice_loss_weight * ice_loss +
#                 self.config.habitat_loss_weight * habitat_loss
#             )
            
#             val_losses['total'] += total_loss.item()
#             val_losses['ice'] += ice_loss.item()
#             val_losses['habitat'] += habitat_loss.item()
        
#         num_batches = len(self.val_loader)
#         for k in val_losses:
#             val_losses[k] /= num_batches
            
#         return val_losses
    
#     def save_checkpoint(self, filename: str, is_best: bool = False):
#         """Save model checkpoint"""
#         checkpoint = {
#             'epoch': self.epoch,
#             'model_state_dict': self.model.state_dict(),
#             'optimizer_state_dict': self.optimizer.state_dict(),
#             'scheduler_state_dict': self.scheduler.state_dict(),
#             'best_val_loss': self.best_val_loss,
#         }
        
#         path = os.path.join(self.config.checkpoint_dir, filename)
#         torch.save(checkpoint, path)
        
#         if is_best:
#             best_path = os.path.join(self.config.checkpoint_dir, 'best_model.pt')
#             torch.save(checkpoint, best_path)
    
#     def load_checkpoint(self, path: str):
#         """Load checkpoint"""
#         checkpoint = torch.load(path, map_location=self.device)
        
#         self.model.load_state_dict(checkpoint['model_state_dict'])
#         self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#         self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
#         self.epoch = checkpoint['epoch']
#         self.best_val_loss = checkpoint['best_val_loss']
        
#         print(f"Loaded checkpoint from epoch {self.epoch}")
    
#     def train(self, resume_path: Optional[str] = None):
#         """Main training loop"""
#         if resume_path and os.path.exists(resume_path):
#             self.load_checkpoint(resume_path)
        
#         print(f"Starting training from epoch {self.epoch}")
#         print(f"Training samples: {len(self.train_loader.dataset)}")
#         print(f"Validation samples: {len(self.val_loader.dataset)}")
        
#         for epoch in range(self.epoch, self.config.epochs):
#             self.epoch = epoch
            
#             # Train
#             train_losses = self.train_epoch()
            
#             # Validate
#             val_losses = self.validate()
            
#             # Scheduler step
#             self.scheduler.step(val_losses['total'])
            
#             # Logging
#             print(f"\nEpoch {epoch}:")
#             print(f"  Train - Total: {train_losses['total']:.4f}, Ice: {train_losses['ice']:.4f}")
#             print(f"  Val   - Total: {val_losses['total']:.4f}, Ice: {val_losses['ice']:.4f}")
            
#             self.writer.add_scalars('Loss/Total', {
#                 'train': train_losses['total'],
#                 'val': val_losses['total']
#             }, epoch)
            
#             self.writer.add_scalars('Loss/Ice', {
#                 'train': train_losses['ice'],
#                 'val': val_losses['ice']
#             }, epoch)
            
#             # Checkpoint
#             if val_losses['total'] < self.best_val_loss:
#                 self.best_val_loss = val_losses['total']
#                 self.patience_counter = 0
#                 self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt', is_best=True)
#                 print(f"  New best model saved!")
#             else:
#                 self.patience_counter += 1
            
#             # Save periodic checkpoint
#             if (epoch + 1) % 10 == 0:
#                 self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt')
            
#             # Early stopping
#             if self.patience_counter >= self.config.early_stopping_patience:
#                 print(f"Early stopping at epoch {epoch}")
#                 break
        
#         print(f"Training complete. Best validation loss: {self.best_val_loss:.4f}")
#         self.writer.close()


# def train_model(
#     data_dir: str,
#     checkpoint_dir: str = './checkpoints',
#     log_dir: str = './logs',
#     device: str = 'cuda',
#     resume_from: Optional[str] = None
# ):
#     """Main training function"""
    
#     # Configs
#     data_config = DataConfig(data_dir=data_dir)
#     model_config = ModelConfig()
#     training_config = TrainingConfig(
#         checkpoint_dir=checkpoint_dir,
#         log_dir=log_dir,
#         device=device
#     )
    
#     # Check device
#     if device == 'cuda' and not torch.cuda.is_available():
#         print("CUDA not available, using CPU")
#         device = 'cpu'
#         training_config.device = 'cpu'
    
#     # Create dataloaders
#     print("Creating dataloaders...")
#     train_loader, val_loader, test_loader = create_dataloaders(
#         data_dir=data_config.data_dir,
#         image_size=data_config.image_size,
#         sequence_length=data_config.sequence_length,
#         prediction_horizon=data_config.prediction_horizon,
#         batch_size=training_config.batch_size,
#         num_workers=training_config.num_workers,
#         train_years=data_config.train_years,
#         val_years=data_config.val_years,
#         test_years=data_config.test_years,
#     )
    
#     # Create model
#     print("Creating model...")
#     model = AntarcticSeaIceForecaster(
#         image_size=data_config.image_size,
#         cnn_channels=model_config.cnn_channels[1:],
#         convlstm_hidden=model_config.convlstm_hidden_dim,
#         convlstm_layers=model_config.convlstm_num_layers,
#         transformer_dim=model_config.transformer_dim,
#         transformer_heads=model_config.transformer_heads,
#         transformer_layers=model_config.transformer_layers,
#         prediction_horizon=data_config.prediction_horizon,
#         use_uncertainty=model_config.use_uncertainty,
#     )
    
#     print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
#     # Create trainer
#     trainer = Trainer(
#         model=model,
#         train_loader=train_loader,
#         val_loader=val_loader,
#         config=training_config,
#         device=device
#     )
    
#     # Train
#     trainer.train(resume_path=resume_from)
    
#     return model, trainer


# if __name__ == "__main__":
#     import argparse
    
#     parser = argparse.ArgumentParser(description='Train Antarctic Sea-Ice Forecaster')
#     parser.add_argument('--data_dir', type=str, required=True, help='Path to NSIDC data')
#     parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
#     parser.add_argument('--log_dir', type=str, default='./logs')
#     parser.add_argument('--device', type=str, default='cuda')
#     parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    
#     args = parser.parse_args()
    
#     train_model(
#         data_dir=args.data_dir,
#         checkpoint_dir=args.checkpoint_dir,
#         log_dir=args.log_dir,
#         device=args.device,
#         resume_from=args.resume
#     )

"""
Training script for Antarctic Sea-Ice Forecasting Model
Includes multi-task loss, uncertainty-aware training, and comprehensive logging
"""
import os
import time
import json
from datetime import datetime
from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import TrainingConfig, DataConfig, ModelConfig
from model import AntarcticSeaIceForecaster, create_model
from data_loader import create_dataloaders


class SeaIceLoss(nn.Module):
    """
    Multi-task loss for sea-ice forecasting
    Combines reconstruction loss, structural similarity, and edge-aware losses
    """
    
    def __init__(self, use_uncertainty: bool = True):
        super().__init__()
        self.use_uncertainty = use_uncertainty
        self.mse = nn.MSELoss(reduction='none')
        self.l1 = nn.L1Loss(reduction='none')
        
    def edge_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Loss on ice edges (melt boundaries)"""
        # Sobel filters for edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=pred.dtype, device=pred.device)
        sobel_y = sobel_x.T
        
        sobel_x = sobel_x.view(1, 1, 3, 3)
        sobel_y = sobel_y.view(1, 1, 3, 3)
        
        # Edge maps
        pred_edges_x = F.conv2d(pred, sobel_x, padding=1)
        pred_edges_y = F.conv2d(pred, sobel_y, padding=1)
        pred_edges = torch.sqrt(pred_edges_x**2 + pred_edges_y**2 + 1e-6)
        
        target_edges_x = F.conv2d(target, sobel_x, padding=1)
        target_edges_y = F.conv2d(target, sobel_y, padding=1)
        target_edges = torch.sqrt(target_edges_x**2 + target_edges_y**2 + 1e-6)
        
        return F.mse_loss(pred_edges, target_edges)
    
    def ssim_loss(self, pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> torch.Tensor:
        """Structural Similarity Index Loss"""
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        # Gaussian window
        sigma = 1.5
        coords = torch.arange(window_size, dtype=pred.dtype, device=pred.device) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        window = g.unsqueeze(0) * g.unsqueeze(1)
        window = window / window.sum()
        window = window.view(1, 1, window_size, window_size)
        
        mu_pred = F.conv2d(pred, window, padding=window_size // 2)
        mu_target = F.conv2d(target, window, padding=window_size // 2)
        
        mu_pred_sq = mu_pred ** 2
        mu_target_sq = mu_target ** 2
        mu_pred_target = mu_pred * mu_target
        
        sigma_pred_sq = F.conv2d(pred ** 2, window, padding=window_size // 2) - mu_pred_sq
        sigma_target_sq = F.conv2d(target ** 2, window, padding=window_size // 2) - mu_target_sq
        sigma_pred_target = F.conv2d(pred * target, window, padding=window_size // 2) - mu_pred_target
        
        ssim = ((2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)) / \
               ((mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2))
        
        return 1 - ssim.mean()
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            predictions: dict with 'ice_maps', 'uncertainty', etc.
            targets: (B, T, 1, H, W) ground truth
        Returns:
            total_loss, loss_dict
        """
        pred_ice = predictions['ice_maps']
        B, T_pred, C, H_pred, W_pred = pred_ice.shape
        
        # Ensure targets match prediction temporal dimension
        T_target = targets.shape[1]
        if T_target > T_pred:
            targets = targets[:, :T_pred]
        elif T_target < T_pred:
            pred_ice = pred_ice[:, :T_target]
            T_pred = T_target
        
        # Resize targets to match prediction spatial size
        B_t, T_t, C_t, H_t, W_t = targets.shape
        if (H_t, W_t) != (H_pred, W_pred):
            # Reshape for interpolation: (B*T, C, H, W)
            targets_flat = targets.view(B_t * T_t, C_t, H_t, W_t)
            targets_resized = F.interpolate(
                targets_flat,
                size=(H_pred, W_pred),
                mode='bilinear',
                align_corners=False
            )
            targets = targets_resized.view(B_t, T_t, C_t, H_pred, W_pred)
        
        losses = {}
        
        # Reconstruction loss (MSE)
        mse_loss = F.mse_loss(pred_ice, targets)
        losses['mse'] = mse_loss.item()
        
        # L1 loss (more robust to outliers)
        l1_loss = F.l1_loss(pred_ice, targets)
        losses['l1'] = l1_loss.item()
        
        # Edge-aware loss
        pred_flat = pred_ice.reshape(-1, 1, H_pred, W_pred)
        target_flat = targets.reshape(-1, 1, H_pred, W_pred)
        edge_loss = self.edge_loss(pred_flat, target_flat)
        losses['edge'] = edge_loss.item()
        
        # SSIM loss
        ssim_loss = self.ssim_loss(pred_flat, target_flat)
        losses['ssim'] = ssim_loss.item()
        
        # Total reconstruction loss
        recon_loss = 0.5 * mse_loss + 0.3 * l1_loss + 0.1 * edge_loss + 0.1 * ssim_loss
        
        # Uncertainty-aware loss (Gaussian NLL)
        if self.use_uncertainty and 'uncertainty' in predictions:
            uncertainty = predictions['uncertainty']
            # Resize uncertainty if needed
            if uncertainty.shape[-2:] != targets.shape[-2:]:
                B_u, T_u, C_u, H_u, W_u = uncertainty.shape
                unc_flat = uncertainty.view(B_u * T_u, C_u, H_u, W_u)
                unc_resized = F.interpolate(
                    unc_flat,
                    size=(H_pred, W_pred),
                    mode='bilinear',
                    align_corners=False
                )
                uncertainty = unc_resized.view(B_u, T_u, C_u, H_pred, W_pred)
            
            variance = uncertainty ** 2 + 1e-6
            
            nll_loss = 0.5 * (torch.log(variance) + (pred_ice - targets) ** 2 / variance)
            nll_loss = nll_loss.mean()
            losses['nll'] = nll_loss.item()
            
            total_loss = recon_loss + 0.1 * nll_loss
        else:
            total_loss = recon_loss
        
        losses['total'] = total_loss.item()
        
        return total_loss, losses


class HabitatRiskLoss(nn.Module):
    """Loss for habitat risk prediction"""
    
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        
    def forward(
        self,
        pred_class: torch.Tensor,
        pred_map: torch.Tensor,
        ice_concentration: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute habitat risk loss based on ice concentration
        Lower ice = higher risk
        """
        # Derive pseudo-labels from ice concentration
        # Mean ice concentration per sample
        mean_ice = ice_concentration.mean(dim=(-2, -1))  # (B, T, 1)
        
        # Bin into 5 risk categories
        risk_labels = torch.zeros_like(mean_ice, dtype=torch.long)
        risk_labels[mean_ice < 0.2] = 4  # Critical risk
        risk_labels[(mean_ice >= 0.2) & (mean_ice < 0.4)] = 3  # High risk
        risk_labels[(mean_ice >= 0.4) & (mean_ice < 0.6)] = 2  # Moderate risk
        risk_labels[(mean_ice >= 0.6) & (mean_ice < 0.8)] = 1  # Low risk
        risk_labels[mean_ice >= 0.8] = 0  # Minimal risk
        
        risk_labels = risk_labels.squeeze(-1)  # (B, T)
        
        # Classification loss
        B, T, num_classes = pred_class.shape
        pred_class_flat = pred_class.view(-1, num_classes)
        risk_labels_flat = risk_labels.view(-1)
        
        class_loss = self.ce(pred_class_flat, risk_labels_flat)
        
        # Spatial consistency: risk map should be high where ice is low
        # Resize ice_concentration to match pred_map size
        B_i, T_i, C_i, H_i, W_i = ice_concentration.shape
        B_p, T_p, C_p, H_p, W_p = pred_map.shape
        
        if (H_i, W_i) != (H_p, W_p):
            # Resize ice concentration to match pred_map
            ice_flat = ice_concentration.view(B_i * T_i, C_i, H_i, W_i)
            ice_resized = F.interpolate(
                ice_flat,
                size=(H_p, W_p),
                mode='bilinear',
                align_corners=False
            )
            ice_concentration_matched = ice_resized.view(B_i, T_i, C_i, H_p, W_p)
        else:
            ice_concentration_matched = ice_concentration
        
        expected_risk_map = 1 - ice_concentration_matched
        map_loss = F.mse_loss(pred_map, expected_risk_map)
        
        return class_loss + 0.5 * map_loss


class Trainer:
    """Main training class"""
    
    def __init__(
        self,
        model: AntarcticSeaIceForecaster,
        train_loader,
        val_loader,
        config: TrainingConfig,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Losses
        self.ice_loss = SeaIceLoss(use_uncertainty=model.use_uncertainty)
        self.habitat_loss = HabitatRiskLoss()
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=config.scheduler_patience
        )
        
        # Logging
        os.makedirs(config.log_dir, exist_ok=True)
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        self.writer = SummaryWriter(config.log_dir)
        
        # State
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_losses = {'total': 0, 'ice': 0, 'habitat': 0}
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.epoch}')
        
        for batch_idx, (inputs, targets, metadata) in enumerate(pbar):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(inputs, forecast_steps=targets.shape[1])
            
            # Compute losses
            ice_loss, ice_losses = self.ice_loss(predictions, targets)
            
            habitat_loss = self.habitat_loss(
                predictions['habitat_risk_class'],
                predictions['habitat_risk_map'],
                predictions['ice_maps']
            )
            
            # Total loss
            total_loss = (
                self.config.ice_loss_weight * ice_loss +
                self.config.habitat_loss_weight * habitat_loss
            )
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip
            )
            
            self.optimizer.step()
            
            # Update metrics
            epoch_losses['total'] += total_loss.item()
            epoch_losses['ice'] += ice_loss.item()
            epoch_losses['habitat'] += habitat_loss.item()
            
            pbar.set_postfix({
                'loss': f"{total_loss.item():.4f}",
                'ice': f"{ice_loss.item():.4f}",
            })
        
        # Average losses
        num_batches = len(self.train_loader)
        for k in epoch_losses:
            epoch_losses[k] /= num_batches
            
        return epoch_losses
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        val_losses = {'total': 0, 'ice': 0, 'habitat': 0}
        
        for inputs, targets, metadata in tqdm(self.val_loader, desc='Validation'):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            predictions = self.model(inputs, forecast_steps=targets.shape[1])
            
            ice_loss, _ = self.ice_loss(predictions, targets)
            habitat_loss = self.habitat_loss(
                predictions['habitat_risk_class'],
                predictions['habitat_risk_map'],
                predictions['ice_maps']
            )
            
            total_loss = (
                self.config.ice_loss_weight * ice_loss +
                self.config.habitat_loss_weight * habitat_loss
            )
            
            val_losses['total'] += total_loss.item()
            val_losses['ice'] += ice_loss.item()
            val_losses['habitat'] += habitat_loss.item()
        
        num_batches = len(self.val_loader)
        for k in val_losses:
            val_losses[k] /= num_batches
            
        return val_losses
    
    def save_checkpoint(self, filename: str, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
        }
        
        path = os.path.join(self.config.checkpoint_dir, filename)
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = os.path.join(self.config.checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, path: str):
        """Load checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"Loaded checkpoint from epoch {self.epoch}")
    
    def train(self, resume_path: Optional[str] = None):
        """Main training loop"""
        if resume_path and os.path.exists(resume_path):
            self.load_checkpoint(resume_path)
        
        print(f"Starting training from epoch {self.epoch}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        
        for epoch in range(self.epoch, self.config.epochs):
            self.epoch = epoch
            
            # Train
            train_losses = self.train_epoch()
            
            # Validate
            val_losses = self.validate()
            
            # Scheduler step
            self.scheduler.step(val_losses['total'])
            
            # Logging
            print(f"\nEpoch {epoch}:")
            print(f"  Train - Total: {train_losses['total']:.4f}, Ice: {train_losses['ice']:.4f}")
            print(f"  Val   - Total: {val_losses['total']:.4f}, Ice: {val_losses['ice']:.4f}")
            
            self.writer.add_scalars('Loss/Total', {
                'train': train_losses['total'],
                'val': val_losses['total']
            }, epoch)
            
            self.writer.add_scalars('Loss/Ice', {
                'train': train_losses['ice'],
                'val': val_losses['ice']
            }, epoch)
            
            # Checkpoint
            if val_losses['total'] < self.best_val_loss:
                self.best_val_loss = val_losses['total']
                self.patience_counter = 0
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt', is_best=True)
                print(f"  New best model saved!")
            else:
                self.patience_counter += 1
            
            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt')
            
            # Early stopping
            if self.patience_counter >= self.config.early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        print(f"Training complete. Best validation loss: {self.best_val_loss:.4f}")
        self.writer.close()


def train_model(
    data_dir: str,
    checkpoint_dir: str = './checkpoints',
    log_dir: str = './logs',
    device: str = 'cuda',
    resume_from: Optional[str] = None
):
    """Main training function"""
    
    # Configs
    data_config = DataConfig(data_dir=data_dir)
    model_config = ModelConfig()
    training_config = TrainingConfig(
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        device=device
    )
    
    # Check device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
        training_config.device = 'cpu'
    
    # Use smaller settings for CPU training
    if device == 'cpu':
        print("Using smaller model configuration for CPU training...")
        data_config.image_size = (64, 64)
        data_config.sequence_length = 30
        data_config.prediction_horizon = 7
        training_config.batch_size = 2
        training_config.num_workers = 0  # Avoid multiprocessing issues on Mac
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=data_config.data_dir,
        image_size=data_config.image_size,
        sequence_length=data_config.sequence_length,
        prediction_horizon=data_config.prediction_horizon,
        batch_size=training_config.batch_size,
        num_workers=training_config.num_workers,
        train_years=data_config.train_years,
        val_years=data_config.val_years,
        test_years=data_config.test_years,
    )
    
    # Create model - use smaller config for CPU
    print("Creating model...")
    if device == 'cpu':
        model = AntarcticSeaIceForecaster(
            image_size=data_config.image_size,
            cnn_channels=(16, 32, 64),  # Smaller
            convlstm_hidden=32,
            convlstm_layers=1,
            transformer_dim=64,
            transformer_heads=4,
            transformer_layers=2,
            prediction_horizon=data_config.prediction_horizon,
            use_uncertainty=True,
        )
    else:
        model = AntarcticSeaIceForecaster(
            image_size=data_config.image_size,
            cnn_channels=model_config.cnn_channels[1:],
            convlstm_hidden=model_config.convlstm_hidden_dim,
            convlstm_layers=model_config.convlstm_num_layers,
            transformer_dim=model_config.transformer_dim,
            transformer_heads=model_config.transformer_heads,
            transformer_layers=model_config.transformer_layers,
            prediction_horizon=data_config.prediction_horizon,
            use_uncertainty=model_config.use_uncertainty,
        )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config,
        device=device
    )
    
    # Train
    trainer.train(resume_path=resume_from)
    
    return model, trainer


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Antarctic Sea-Ice Forecaster')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to NSIDC data')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    train_model(
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        device=args.device,
        resume_from=args.resume
    )