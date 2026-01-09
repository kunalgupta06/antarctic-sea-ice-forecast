"""
Antarctic Sea-Ice Forecasting - Interactive Notebook
=====================================================

This script is designed for Jupyter/Colab experimentation.
Run cells sequentially or import as a module.

Usage in Jupyter:
    %run notebook_demo.py
    
Or import:
    from notebook_demo import quick_train, quick_forecast
"""

#%% [markdown]
# # 🐧 Antarctic Sea-Ice Forecasting Demo
# 
# This notebook demonstrates the complete pipeline for predicting
# Antarctic sea-ice 50 years into the future.

#%% Cell 1: Imports and Setup
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Check GPU
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

#%% Cell 2: Import Project Modules
from model import AntarcticSeaIceForecaster
from data_loader import NSIDCDataLoader, SeaIceSequenceDataset

#%% Cell 3: Configuration
class QuickConfig:
    """Lightweight config for demos"""
    # Data
    data_dir = "./data/nsidc_images"  # Change this to your data path
    image_size = (128, 128)  # Smaller for faster demo
    sequence_length = 30  # Days of history
    prediction_horizon = 12  # Days to predict
    
    # Model
    cnn_channels = (32, 64, 128)
    convlstm_hidden = 64
    transformer_layers = 2
    
    # Training
    batch_size = 4
    epochs = 10
    learning_rate = 1e-3

config = QuickConfig()

#%% Cell 4: Create Model
def create_demo_model(config):
    """Create a smaller model for demo purposes"""
    model = AntarcticSeaIceForecaster(
        image_size=config.image_size,
        cnn_channels=config.cnn_channels,
        convlstm_hidden=config.convlstm_hidden,
        convlstm_layers=1,
        transformer_dim=128,
        transformer_heads=4,
        transformer_layers=config.transformer_layers,
        prediction_horizon=config.prediction_horizon,
        use_uncertainty=True
    )
    
    # Print model info
    params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {params:,}")
    
    return model.to(DEVICE)

# Create model
model = create_demo_model(config)

#%% Cell 5: Test Forward Pass
def test_forward_pass(model, config):
    """Test that the model works"""
    print("Testing forward pass...")
    
    # Create dummy input
    batch_size = 2
    x = torch.randn(
        batch_size, 
        config.sequence_length, 
        1, 
        config.image_size[0], 
        config.image_size[1]
    ).to(DEVICE)
    
    # Forward pass
    with torch.no_grad():
        output = model(x, forecast_steps=config.prediction_horizon)
    
    print(f"Input shape: {x.shape}")
    print(f"Output ice_maps: {output['ice_maps'].shape}")
    print(f"Output habitat_risk: {output['habitat_risk_class'].shape}")
    print(f"Output uncertainty: {output['uncertainty'].shape}")
    
    return output

output = test_forward_pass(model, config)

#%% Cell 6: Visualize Sample Output
def visualize_sample_output(output):
    """Visualize model outputs"""
    ice_maps = output['ice_maps'][0].cpu().numpy()  # First batch
    uncertainty = output['uncertainty'][0].cpu().numpy()
    risk = output['habitat_risk_class'][0].cpu().numpy()
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Ice maps at different timesteps
    for i, t in enumerate([0, 3, 6, 11]):
        if t < ice_maps.shape[0]:
            axes[0, i].imshow(ice_maps[t, 0], cmap='Blues_r', vmin=0, vmax=1)
            axes[0, i].set_title(f'Ice (t+{t+1})')
            axes[0, i].axis('off')
            
            axes[1, i].imshow(uncertainty[t, 0], cmap='Reds', vmin=0)
            axes[1, i].set_title(f'Uncertainty (t+{t+1})')
            axes[1, i].axis('off')
    
    plt.suptitle('Sample Model Output')
    plt.tight_layout()
    plt.show()
    
    # Risk progression
    fig, ax = plt.subplots(figsize=(10, 4))
    risk_labels = ['Minimal', 'Low', 'Moderate', 'High', 'Critical']
    
    for i, label in enumerate(risk_labels):
        ax.plot(risk[:, i], label=label)
    
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Risk Probability')
    ax.set_title('Habitat Risk Progression')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.show()

visualize_sample_output(output)

#%% Cell 7: Quick Training Function
def quick_train(model, config, synthetic=True, epochs=None):
    """
    Quick training loop for demo
    
    Args:
        model: The forecaster model
        config: Configuration object
        synthetic: If True, use synthetic data
        epochs: Override number of epochs
    """
    from torch.utils.data import DataLoader, TensorDataset
    import torch.nn.functional as F
    
    if epochs is None:
        epochs = config.epochs
    
    # Create synthetic or real data
    if synthetic:
        print("Using synthetic data for demo...")
        n_samples = 100
        X = torch.randn(n_samples, config.sequence_length, 1, *config.image_size)
        Y = torch.randn(n_samples, config.prediction_horizon, 1, *config.image_size).sigmoid()
        
        dataset = TensorDataset(X, Y)
        loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    else:
        print(f"Loading data from {config.data_dir}...")
        nsidc_loader = NSIDCDataLoader(config.data_dir, config.image_size)
        dataset = SeaIceSequenceDataset(
            nsidc_loader,
            sequence_length=config.sequence_length,
            prediction_horizon=config.prediction_horizon
        )
        loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    # Training loop
    model.train()
    losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        n_batches = 0
        
        for batch in loader:
            if synthetic:
                inputs, targets = batch
            else:
                inputs, targets, _ = batch
            
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)
            
            optimizer.zero_grad()
            
            output = model(inputs, forecast_steps=targets.shape[1])
            
            # Simple MSE loss for demo
            pred = output['ice_maps']
            if pred.shape[-2:] != targets.shape[-2:]:
                targets = F.interpolate(
                    targets.view(-1, 1, *targets.shape[-2:]),
                    size=pred.shape[-2:],
                    mode='bilinear'
                ).view(*targets.shape[:3], *pred.shape[-2:])
            
            loss = F.mse_loss(pred, targets)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
    
    # Plot training curve
    plt.figure(figsize=(10, 4))
    plt.plot(losses, 'b-o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return losses

# Train for a few epochs
print("\n" + "="*50)
print("Quick Training Demo")
print("="*50)
losses = quick_train(model, config, synthetic=True, epochs=5)

#%% Cell 8: Generate Forecast
def quick_forecast(model, config, years=5):
    """Generate a quick multi-year forecast"""
    print(f"\nGenerating {years}-year forecast...")
    
    model.eval()
    
    # Initial sequence (random for demo)
    initial = torch.randn(1, config.sequence_length, 1, *config.image_size).to(DEVICE)
    
    all_predictions = []
    current_input = initial
    samples_per_year = 12
    
    with torch.no_grad():
        for year in range(years):
            output = model(current_input, forecast_steps=samples_per_year)
            predictions = output['ice_maps']
            
            all_predictions.append(predictions.cpu())
            
            # Update input
            current_input = torch.cat([
                current_input[:, samples_per_year:],
                predictions
            ], dim=1)
    
    # Concatenate all years
    forecast = torch.cat(all_predictions, dim=1)  # (1, years*12, 1, H, W)
    
    print(f"Forecast shape: {forecast.shape}")
    
    # Compute annual extent
    extent = (forecast > 0.15).float().mean(dim=(-2, -1)).squeeze()
    annual_extent = extent.view(years, samples_per_year).mean(dim=1)
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Extent over time
    ax = axes[0, 0]
    ax.plot(extent.numpy(), 'b-', alpha=0.7)
    ax.set_xlabel('Months')
    ax.set_ylabel('Ice Extent (fraction > 15%)')
    ax.set_title('Monthly Sea Ice Extent')
    ax.grid(True, alpha=0.3)
    
    # Annual average
    ax = axes[0, 1]
    ax.bar(range(1, years+1), annual_extent.numpy(), color='steelblue')
    ax.set_xlabel('Year')
    ax.set_ylabel('Mean Ice Extent')
    ax.set_title('Annual Average Ice Extent')
    ax.grid(True, alpha=0.3)
    
    # Sample maps
    ax = axes[1, 0]
    ax.imshow(forecast[0, 0, 0].numpy(), cmap='Blues_r', vmin=0, vmax=1)
    ax.set_title('Year 1, Month 1')
    ax.axis('off')
    
    ax = axes[1, 1]
    ax.imshow(forecast[0, -1, 0].numpy(), cmap='Blues_r', vmin=0, vmax=1)
    ax.set_title(f'Year {years}, Month 12')
    ax.axis('off')
    
    plt.suptitle(f'{years}-Year Sea Ice Forecast', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    return forecast, annual_extent

forecast, annual_extent = quick_forecast(model, config, years=5)

#%% Cell 9: Monte Carlo Uncertainty
def mc_uncertainty_demo(model, config, mc_samples=5, years=3):
    """Demonstrate Monte Carlo dropout uncertainty"""
    print(f"\nMonte Carlo Uncertainty Demo ({mc_samples} samples, {years} years)")
    
    # Enable dropout during inference
    def enable_dropout(model):
        for m in model.modules():
            if isinstance(m, torch.nn.Dropout) or isinstance(m, torch.nn.Dropout2d):
                m.train()
    
    initial = torch.randn(1, config.sequence_length, 1, *config.image_size).to(DEVICE)
    
    all_forecasts = []
    
    for i in range(mc_samples):
        model.eval()
        enable_dropout(model)
        
        forecast, _ = quick_forecast.__wrapped__(model, config, years) \
            if hasattr(quick_forecast, '__wrapped__') else (
                # Simplified inline forecast
                _single_forecast(model, initial, years, config)
            )
        all_forecasts.append(forecast)
    
    # Stack and compute stats
    stacked = torch.stack(all_forecasts)  # (samples, 1, time, 1, H, W)
    
    mean_forecast = stacked.mean(dim=0)
    std_forecast = stacked.std(dim=0)
    
    # Visualize uncertainty growth
    spatial_std = std_forecast.mean(dim=(-2, -1)).squeeze()
    
    plt.figure(figsize=(12, 4))
    plt.plot(spatial_std.numpy(), 'r-', linewidth=2)
    plt.fill_between(range(len(spatial_std)), 0, spatial_std.numpy(), alpha=0.3, color='red')
    plt.xlabel('Months')
    plt.ylabel('Prediction Uncertainty (std)')
    plt.title('Uncertainty Growth Over Forecast Horizon')
    plt.grid(True, alpha=0.3)
    
    for y in range(years):
        plt.axvline(y * 12, color='gray', linestyle='--', alpha=0.5)
    
    plt.show()
    
    return mean_forecast, std_forecast

def _single_forecast(model, initial, years, config):
    """Helper for single forecast run"""
    all_preds = []
    current = initial
    
    with torch.no_grad():
        for _ in range(years):
            out = model(current, forecast_steps=12)
            preds = out['ice_maps']
            all_preds.append(preds.cpu())
            current = torch.cat([current[:, 12:], preds], dim=1)
    
    return torch.cat(all_preds, dim=1), None

# Run MC uncertainty demo
mean_fc, std_fc = mc_uncertainty_demo(model, config, mc_samples=3, years=3)

#%% Cell 10: Save Model
def save_demo_model(model, path="demo_model.pt"):
    """Save the trained model"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'image_size': config.image_size,
            'cnn_channels': config.cnn_channels,
            'convlstm_hidden': config.convlstm_hidden,
        },
        'timestamp': datetime.now().isoformat()
    }, path)
    print(f"Model saved to {path}")

# Uncomment to save:
# save_demo_model(model)

#%% [markdown]
# ## 🎯 Next Steps
# 
# 1. **Download real NSIDC data** from the link in README
# 2. **Update `config.data_dir`** to point to your data
# 3. **Set `synthetic=False`** in `quick_train()`
# 4. **Increase epochs** for better training
# 5. **Use `forecast.py`** for full 50-year predictions
# 
# Good luck at the hackathon! 🚀

print("\n" + "="*50)
print("Demo Complete! 🎉")
print("="*50)
print("""
Next steps:
1. Download real data from NSIDC
2. Run: python main.py --mode train --data_dir /your/data/path
3. Run: python main.py --mode forecast --model_path checkpoints/best_model.pt
""")