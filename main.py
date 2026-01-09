"""
Antarctic Sea-Ice Forecasting System - Main Entry Point
========================================================

A deep learning system for:
1. Predicting Antarctic sea-ice concentration from satellite imagery
2. Forecasting 50 years into the future
3. Estimating penguin habitat risk
4. Quantifying prediction uncertainty

Usage:
    python main.py --data_dir /path/to/nsidc/data --mode train
    python main.py --data_dir /path/to/nsidc/data --mode forecast --model_path checkpoints/best_model.pt
"""

import os
import sys
import argparse
from datetime import datetime
import torch

# Project imports
from config import DataConfig, ModelConfig, TrainingConfig, ForecastConfig
from data_loader import NSIDCDataLoader, create_dataloaders
from model import AntarcticSeaIceForecaster
from train import train_model, Trainer
from forecast import generate_50_year_forecast, LongTermForecaster


def print_banner():
    """Print project banner"""
    banner = """
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                                                                   ║
    ║       🐧 Antarctic Sea-Ice Forecasting System 🧊                  ║
    ║                                                                   ║
    ║   Deep Learning for Climate Prediction & Ecosystem Protection    ║
    ║                                                                   ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def check_data(data_dir: str) -> bool:
    """Check if data directory has valid images"""
    print("\n📂 Checking data directory...")
    
    if not os.path.exists(data_dir):
        print(f"   ❌ Directory not found: {data_dir}")
        print("   Please download data from: https://noaadata.apps.nsidc.org/NOAA/G02135/south/daily/images/")
        return False
    
    loader = NSIDCDataLoader(data_dir, image_size=(256, 256))
    dates = loader.get_available_dates()
    
    if len(dates) == 0:
        print(f"   ❌ No valid images found in {data_dir}")
        print("   Expected formats: S_YYYYMMDD_*.png or similar dated images")
        return False
    
    date_range = loader.get_date_range()
    print(f"   ✅ Found {len(dates)} images")
    print(f"   📅 Date range: {date_range[0]} to {date_range[1]}")
    
    # Test loading
    sample = loader.load_image(dates[0])
    if sample is not None:
        print(f"   📐 Image size: {sample.shape}")
        print(f"   📊 Value range: [{sample.min():.3f}, {sample.max():.3f}]")
    
    return True


def run_training(args):
    """Run model training"""
    print("\n🏋️ Starting Training Pipeline")
    print("=" * 50)
    
    # Validate data
    if not check_data(args.data_dir):
        sys.exit(1)
    
    # Set device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, switching to CPU")
        device = 'cpu'
    print(f"   🖥️  Device: {device}")
    
    # Training
    model, trainer = train_model(
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        device=device,
        resume_from=args.resume
    )
    
    print("\n✅ Training complete!")
    print(f"   Best model saved to: {args.checkpoint_dir}/best_model.pt")


def run_forecast(args):
    """Run 50-year forecasting"""
    print("\n🔮 Starting 50-Year Forecast")
    print("=" * 50)
    
    if not os.path.exists(args.model_path):
        print(f"❌ Model not found: {args.model_path}")
        print("   Please train a model first or provide valid path")
        sys.exit(1)
    
    results = generate_50_year_forecast(
        model_path=args.model_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=args.device,
        ensemble_size=args.ensemble_size
    )
    
    print("\n✅ Forecast complete!")


def run_demo(args):
    """Run a quick demo with synthetic data"""
    print("\n🎮 Running Demo Mode")
    print("=" * 50)
    print("   (Using synthetic data for demonstration)")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create a SMALLER model for demo (original was too large)
    print("\n1. Creating model...")
    model = AntarcticSeaIceForecaster(
        image_size=(64, 64),  # Smaller images
        cnn_channels=(16, 32, 64),  # Fewer channels
        convlstm_hidden=32,
        convlstm_layers=1,
        transformer_dim=64,
        transformer_heads=4,
        transformer_layers=2,
        prediction_horizon=12
    )
    model = model.to(device)
    
    params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {params:,}")
    
    # Synthetic input - smaller batch and sequence
    print("\n2. Testing forward pass...")
    batch = torch.randn(2, 20, 1, 64, 64).to(device)
    
    with torch.no_grad():
        output = model(batch, forecast_steps=12)
    
    print(f"   Input shape: {batch.shape}")
    print(f"   Output ice maps: {output['ice_maps'].shape}")
    print(f"   Output habitat risk: {output['habitat_risk_class'].shape}")
    
    # Test long-term forecast
    print("\n3. Testing 3-year forecast...")
    forecaster = LongTermForecaster(model, device=device, ensemble_size=2)
    
    demo_input = torch.randn(1, 20, 1, 64, 64).to(device)
    
    # Quick forecast (3 years instead of 50 for demo)
    model.eval()
    current_input = demo_input
    all_predictions = []
    
    with torch.no_grad():
        for year in range(3):
            output = model(current_input, forecast_steps=12)
            all_predictions.append(output['ice_maps'].cpu())
            
            # Update input
            predicted = output['ice_maps']
            if current_input.shape[1] > 12:
                current_input = torch.cat([current_input[:, 12:], predicted], dim=1)
            else:
                current_input = predicted
    
    forecast = torch.cat(all_predictions, dim=1)
    print(f"   Generated {forecast.shape[1]} monthly predictions (3 years)")
    
    # Compute ice extent trend
    extent = (forecast > 0.5).float().mean(dim=(-2, -1)).squeeze()
    print(f"   Ice extent trend: {extent[:6].numpy().round(3)}...")
    
    print("\n✅ Demo complete! Model is working correctly.")
    print("   Train on real NSIDC data for actual predictions.")
    print("\n   Next steps:")
    print("   1. Point --data_dir to your NSIDC images folder")
    print("   2. Run: python3 main.py --mode train --data_dir /your/data/path")


def run_evaluate(args):
    """Evaluate model on test set"""
    print("\n📊 Running Evaluation")
    print("=" * 50)
    
    if not os.path.exists(args.model_path):
        print(f"❌ Model not found: {args.model_path}")
        sys.exit(1)
    
    if not check_data(args.data_dir):
        sys.exit(1)
    
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    
    # Load model
    print("\n1. Loading model...")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    model = AntarcticSeaIceForecaster(image_size=(256, 256))
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Create test dataloader
    print("\n2. Loading test data...")
    _, _, test_loader = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=4,
        num_workers=2
    )
    
    print(f"   Test samples: {len(test_loader.dataset)}")
    
    # Evaluate
    print("\n3. Evaluating...")
    from train import SeaIceLoss
    
    loss_fn = SeaIceLoss()
    total_loss = 0
    total_mse = 0
    n_batches = 0
    
    with torch.no_grad():
        for inputs, targets, _ in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs, forecast_steps=targets.shape[1])
            loss, losses = loss_fn(outputs, targets)
            
            total_loss += loss.item()
            total_mse += losses['mse']
            n_batches += 1
    
    print(f"\n   📈 Results:")
    print(f"      Total Loss: {total_loss / n_batches:.4f}")
    print(f"      MSE: {total_mse / n_batches:.4f}")
    print(f"      RMSE: {(total_mse / n_batches) ** 0.5:.4f}")


def main():
    """Main entry point"""
    print_banner()
    
    parser = argparse.ArgumentParser(
        description='Antarctic Sea-Ice Forecasting System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check your data
  python main.py --mode check --data_dir ./data/nsidc_images
  
  # Train the model
  python main.py --mode train --data_dir ./data/nsidc_images
  
  # Generate 50-year forecast
  python main.py --mode forecast --data_dir ./data/nsidc_images --model_path ./checkpoints/best_model.pt
  
  # Run demo with synthetic data
  python main.py --mode demo
        """
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['train', 'forecast', 'evaluate', 'demo', 'check'],
        help='Mode: train, forecast, evaluate, demo, or check'
    )
    
    parser.add_argument(
        '--data_dir',
        type=str,
        default='./data/nsidc_images',
        help='Path to NSIDC image data'
    )
    
    parser.add_argument(
        '--model_path',
        type=str,
        default='./checkpoints/best_model.pt',
        help='Path to trained model checkpoint'
    )
    
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default='./checkpoints',
        help='Directory for saving checkpoints'
    )
    
    parser.add_argument(
        '--log_dir',
        type=str,
        default='./logs',
        help='Directory for TensorBoard logs'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./forecast_results',
        help='Directory for forecast outputs'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use'
    )
    
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Resume training from checkpoint'
    )
    
    parser.add_argument(
        '--ensemble_size',
        type=int,
        default=10,
        help='Number of ensemble members for uncertainty'
    )
    
    args = parser.parse_args()
    
    # Dispatch to appropriate function
    if args.mode == 'train':
        run_training(args)
    elif args.mode == 'forecast':
        run_forecast(args)
    elif args.mode == 'evaluate':
        run_evaluate(args)
    elif args.mode == 'demo':
        run_demo(args)
    elif args.mode == 'check':
        check_data(args.data_dir)


if __name__ == "__main__":
    main()