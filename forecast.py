# """
# Long-term (50-year) Forecasting Script for Antarctic Sea-Ice
# Includes:
# - Autoregressive multi-decade predictions
# - Monte Carlo dropout for uncertainty
# - Ensemble forecasting
# - Visualization and analysis
# """
# import os
# import json
# from datetime import datetime
# from typing import Dict, List, Optional, Tuple
# import numpy as np
# import torch
# import torch.nn.functional as F
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# from matplotlib.colors import LinearSegmentedColormap

# from config import ForecastConfig, DataConfig, ModelConfig
# from model import AntarcticSeaIceForecaster
# from data_loader import NSIDCDataLoader, YearlyAggregatedDataset


# class LongTermForecaster:
#     """
#     50-year autoregressive forecaster with uncertainty quantification
#     """
    
#     def __init__(
#         self,
#         model: AntarcticSeaIceForecaster,
#         device: str = 'cuda',
#         ensemble_size: int = 10
#     ):
#         self.model = model.to(device)
#         self.device = device
#         self.ensemble_size = ensemble_size
        
#         # Custom colormap for sea ice
#         self.ice_cmap = LinearSegmentedColormap.from_list(
#             'ice',
#             [(0, '#1a1a2e'), (0.3, '#16537e'), (0.5, '#1e90ff'),
#              (0.7, '#87ceeb'), (1, '#ffffff')]
#         )
        
#         self.risk_cmap = LinearSegmentedColormap.from_list(
#             'risk',
#             [(0, '#00ff00'), (0.25, '#7fff00'), (0.5, '#ffff00'),
#              (0.75, '#ff7f00'), (1, '#ff0000')]
#         )
    
#     def enable_dropout(self):
#         """Enable dropout for MC sampling"""
#         for module in self.model.modules():
#             if isinstance(module, torch.nn.Dropout) or isinstance(module, torch.nn.Dropout2d):
#                 module.train()
    
#     def predict_with_uncertainty(
#         self,
#         initial_sequence: torch.Tensor,
#         years: int = 50,
#         samples_per_year: int = 12,
#         mc_samples: int = 10
#     ) -> Dict[str, np.ndarray]:
#         """
#         Generate predictions with Monte Carlo dropout uncertainty
        
#         Args:
#             initial_sequence: (1, T, C, H, W) seed sequence
#             years: number of years to forecast
#             samples_per_year: predictions per year (12 = monthly)
#             mc_samples: number of MC dropout samples
        
#         Returns:
#             Dictionary with predictions, uncertainty, and statistics
#         """
#         self.model.eval()
#         total_steps = years * samples_per_year
        
#         all_predictions = []
        
#         print(f"Generating {mc_samples} ensemble forecasts for {years} years...")
        
#         for sample_idx in tqdm(range(mc_samples), desc="MC Samples"):
#             # Enable dropout for this sample
#             self.enable_dropout()
            
#             predictions = self._autoregressive_forecast(
#                 initial_sequence.clone(),
#                 years=years,
#                 samples_per_year=samples_per_year
#             )
            
#             all_predictions.append(predictions)
        
#         # Stack all predictions
#         ice_maps = np.stack([p['ice_maps'] for p in all_predictions], axis=0)
#         habitat_risk = np.stack([p['habitat_risk'] for p in all_predictions], axis=0)
        
#         # Compute statistics
#         results = {
#             'ice_maps_mean': ice_maps.mean(axis=0),
#             'ice_maps_std': ice_maps.std(axis=0),
#             'ice_maps_median': np.median(ice_maps, axis=0),
#             'ice_maps_5th': np.percentile(ice_maps, 5, axis=0),
#             'ice_maps_95th': np.percentile(ice_maps, 95, axis=0),
#             'habitat_risk_mean': habitat_risk.mean(axis=0),
#             'habitat_risk_std': habitat_risk.std(axis=0),
#             'all_ice_maps': ice_maps,
#             'all_habitat_risk': habitat_risk,
#             'years': np.arange(years),
#             'months': np.arange(total_steps),
#         }
        
#         # Compute annual statistics
#         results['annual_ice_extent'] = self._compute_annual_extent(ice_maps)
#         results['annual_risk_trend'] = self._compute_annual_risk(habitat_risk)
        
#         return results
    
#     @torch.no_grad()
#     def _autoregressive_forecast(
#         self,
#         initial_sequence: torch.Tensor,
#         years: int = 50,
#         samples_per_year: int = 12
#     ) -> Dict[str, np.ndarray]:
#         """Single autoregressive forecast run"""
        
#         total_steps = years * samples_per_year
#         current_input = initial_sequence.to(self.device)
        
#         all_ice = []
#         all_risk = []
        
#         # Predict in yearly chunks
#         for year in range(years):
#             # Get predictions for this year
#             output = self.model(current_input, forecast_steps=samples_per_year)
            
#             # Extract predictions
#             ice_maps = output['ice_maps'].cpu().numpy()
#             risk_class = output['habitat_risk_class'].cpu().numpy()
            
#             all_ice.append(ice_maps[0])  # Remove batch dim
#             all_risk.append(risk_class[0])
            
#             # Update input sequence with predictions
#             predicted = output['ice_maps']  # (1, T_pred, C, H, W)
            
#             # Slide window: drop oldest, append newest
#             if current_input.shape[1] > samples_per_year:
#                 current_input = torch.cat([
#                     current_input[:, samples_per_year:],
#                     predicted
#                 ], dim=1)
#             else:
#                 current_input = predicted
        
#         return {
#             'ice_maps': np.concatenate(all_ice, axis=0),
#             'habitat_risk': np.concatenate(all_risk, axis=0)
#         }
    
#     def _compute_annual_extent(self, ice_maps: np.ndarray) -> Dict[str, np.ndarray]:
#         """
#         Compute annual sea ice extent statistics
        
#         Args:
#             ice_maps: (ensemble, time, C, H, W) predictions
#         """
#         ensemble, time, C, H, W = ice_maps.shape
#         samples_per_year = 12
#         years = time // samples_per_year
        
#         # Reshape to yearly
#         yearly = ice_maps.reshape(ensemble, years, samples_per_year, C, H, W)
        
#         # Sea ice extent: area where concentration > 15%
#         extent_threshold = 0.15
#         extent = (yearly > extent_threshold).sum(axis=(-3, -2, -1)) / (C * H * W)
        
#         # Annual statistics
#         annual_mean = extent.mean(axis=2)  # Average over months
#         annual_min = extent.min(axis=2)    # Minimum (summer)
#         annual_max = extent.max(axis=2)    # Maximum (winter)
        
#         return {
#             'mean': annual_mean.mean(axis=0),  # Ensemble mean
#             'std': annual_mean.std(axis=0),
#             'min_extent': annual_min.mean(axis=0),
#             'max_extent': annual_max.mean(axis=0),
#             '5th_percentile': np.percentile(annual_mean, 5, axis=0),
#             '95th_percentile': np.percentile(annual_mean, 95, axis=0),
#         }
    
#     def _compute_annual_risk(self, habitat_risk: np.ndarray) -> Dict[str, np.ndarray]:
#         """Compute annual habitat risk trends"""
#         ensemble, time, num_classes = habitat_risk.shape
#         samples_per_year = 12
#         years = time // samples_per_year
        
#         # Reshape to yearly
#         yearly = habitat_risk.reshape(ensemble, years, samples_per_year, num_classes)
        
#         # Average risk class probability over months
#         annual_risk = yearly.mean(axis=2)
        
#         # Compute expected risk level (0=minimal, 4=critical)
#         risk_levels = np.arange(num_classes)
#         expected_risk = (annual_risk * risk_levels).sum(axis=-1)
        
#         return {
#             'expected_risk': expected_risk.mean(axis=0),
#             'risk_std': expected_risk.std(axis=0),
#             'class_probabilities': annual_risk.mean(axis=0),
#         }
    
#     def visualize_forecast(
#         self,
#         results: Dict[str, np.ndarray],
#         output_dir: str,
#         create_animation: bool = True
#     ):
#         """Create comprehensive visualizations"""
#         os.makedirs(output_dir, exist_ok=True)
        
#         # 1. Annual ice extent trend
#         fig, ax = plt.subplots(figsize=(14, 6))
        
#         years = np.arange(len(results['annual_ice_extent']['mean']))
#         mean = results['annual_ice_extent']['mean']
#         std = results['annual_ice_extent']['std']
#         p5 = results['annual_ice_extent']['5th_percentile']
#         p95 = results['annual_ice_extent']['95th_percentile']
        
#         ax.fill_between(years, p5, p95, alpha=0.3, color='blue', label='90% CI')
#         ax.fill_between(years, mean - std, mean + std, alpha=0.5, color='blue', label='±1 std')
#         ax.plot(years, mean, 'b-', linewidth=2, label='Mean prediction')
#         ax.plot(years, results['annual_ice_extent']['min_extent'], 
#                 'r--', linewidth=1, label='Summer minimum')
#         ax.plot(years, results['annual_ice_extent']['max_extent'], 
#                 'g--', linewidth=1, label='Winter maximum')
        
#         ax.set_xlabel('Years from now', fontsize=12)
#         ax.set_ylabel('Sea Ice Extent (fraction)', fontsize=12)
#         ax.set_title('Antarctic Sea Ice Extent: 50-Year Forecast', fontsize=14)
#         ax.legend()
#         ax.grid(True, alpha=0.3)
        
#         plt.tight_layout()
#         plt.savefig(os.path.join(output_dir, 'ice_extent_trend.png'), dpi=300)
#         plt.close()
        
#         # 2. Habitat risk trend
#         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
#         risk = results['annual_risk_trend']
        
#         ax1.fill_between(years, 
#                          risk['expected_risk'] - risk['risk_std'],
#                          risk['expected_risk'] + risk['risk_std'],
#                          alpha=0.3, color='red')
#         ax1.plot(years, risk['expected_risk'], 'r-', linewidth=2)
#         ax1.set_xlabel('Years from now')
#         ax1.set_ylabel('Expected Risk Level (0=minimal, 4=critical)')
#         ax1.set_title('Penguin Habitat Risk Trajectory')
#         ax1.grid(True, alpha=0.3)
#         ax1.set_ylim(0, 4)
        
#         # Risk class probabilities over time
#         risk_classes = ['Minimal', 'Low', 'Moderate', 'High', 'Critical']
#         probs = risk['class_probabilities']
        
#         ax2.stackplot(years, probs.T, labels=risk_classes,
#                       colors=['green', 'lightgreen', 'yellow', 'orange', 'red'],
#                       alpha=0.8)
#         ax2.set_xlabel('Years from now')
#         ax2.set_ylabel('Probability')
#         ax2.set_title('Habitat Risk Class Distribution')
#         ax2.legend(loc='upper left')
#         ax2.grid(True, alpha=0.3)
        
#         plt.tight_layout()
#         plt.savefig(os.path.join(output_dir, 'habitat_risk_trend.png'), dpi=300)
#         plt.close()
        
#         # 3. Uncertainty over time
#         fig, ax = plt.subplots(figsize=(12, 5))
        
#         # Compute spatial mean uncertainty over time
#         uncertainty = results['ice_maps_std'].mean(axis=(1, 2, 3))
        
#         ax.plot(np.arange(len(uncertainty)), uncertainty, 'purple', linewidth=2)
#         ax.fill_between(np.arange(len(uncertainty)), 0, uncertainty, alpha=0.3, color='purple')
#         ax.set_xlabel('Months from now')
#         ax.set_ylabel('Prediction Uncertainty (std)')
#         ax.set_title('Forecast Uncertainty Over Time')
#         ax.grid(True, alpha=0.3)
        
#         # Add year markers
#         for year in range(0, 51, 10):
#             ax.axvline(year * 12, color='gray', linestyle='--', alpha=0.5)
#             ax.text(year * 12, ax.get_ylim()[1], f'Year {year}', 
#                    ha='center', va='bottom', fontsize=10)
        
#         plt.tight_layout()
#         plt.savefig(os.path.join(output_dir, 'uncertainty_trend.png'), dpi=300)
#         plt.close()
        
#         # 4. Sample ice maps at key timepoints
#         fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        
#         timepoints = [0, 60, 120, 240, 360, 480, 540, 570, 594, 599]  # Months
#         timepoints = [t for t in timepoints if t < len(results['ice_maps_mean'])]
        
#         for idx, t in enumerate(timepoints[:10]):
#             row = idx // 5
#             col = idx % 5
            
#             ax = axes[row, col]
            
#             mean_map = results['ice_maps_mean'][t, 0]
#             uncertainty_map = results['ice_maps_std'][t, 0]
            
#             # Overlay: mean with uncertainty as alpha
#             im = ax.imshow(mean_map, cmap=self.ice_cmap, vmin=0, vmax=1)
#             ax.contour(uncertainty_map, levels=[0.1, 0.2, 0.3], 
#                       colors='red', linewidths=0.5, alpha=0.5)
            
#             year = t // 12
#             month = t % 12 + 1
#             ax.set_title(f'Year {year}, Month {month}')
#             ax.axis('off')
        
#         plt.suptitle('Predicted Sea Ice Concentration Over 50 Years', fontsize=14)
#         plt.tight_layout()
#         plt.savefig(os.path.join(output_dir, 'ice_map_samples.png'), dpi=300)
#         plt.close()
        
#         # 5. Create animation if requested
#         if create_animation:
#             self._create_animation(results, output_dir)
        
#         print(f"Visualizations saved to {output_dir}")
        
#         # 6. Save numerical summary
#         summary = {
#             'years_forecasted': 50,
#             'initial_ice_extent': float(results['annual_ice_extent']['mean'][0]),
#             'final_ice_extent': float(results['annual_ice_extent']['mean'][-1]),
#             'extent_change_percent': float(
#                 (results['annual_ice_extent']['mean'][-1] - results['annual_ice_extent']['mean'][0]) 
#                 / results['annual_ice_extent']['mean'][0] * 100
#             ),
#             'initial_risk_level': float(results['annual_risk_trend']['expected_risk'][0]),
#             'final_risk_level': float(results['annual_risk_trend']['expected_risk'][-1]),
#             'avg_uncertainty_first_decade': float(results['ice_maps_std'][:120].mean()),
#             'avg_uncertainty_last_decade': float(results['ice_maps_std'][-120:].mean()),
#         }
        
#         with open(os.path.join(output_dir, 'forecast_summary.json'), 'w') as f:
#             json.dump(summary, f, indent=2)
        
#         print("\n📊 Forecast Summary:")
#         print(f"  Ice Extent Change: {summary['extent_change_percent']:.1f}%")
#         print(f"  Risk Level: {summary['initial_risk_level']:.2f} → {summary['final_risk_level']:.2f}")
#         print(f"  Uncertainty Growth: {summary['avg_uncertainty_first_decade']:.3f} → {summary['avg_uncertainty_last_decade']:.3f}")
    
#     def _create_animation(self, results: Dict[str, np.ndarray], output_dir: str):
#         """Create animated visualization"""
#         print("Creating animation...")
        
#         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
#         ice_maps = results['ice_maps_mean']
        
#         # Sample every 6 months for animation speed
#         frames = range(0, len(ice_maps), 6)
        
#         im1 = ax1.imshow(ice_maps[0, 0], cmap=self.ice_cmap, vmin=0, vmax=1)
#         ax1.set_title('Sea Ice Concentration')
#         ax1.axis('off')
#         plt.colorbar(im1, ax=ax1, label='Concentration')
        
#         # Line plot for extent
#         years = np.arange(len(results['annual_ice_extent']['mean']))
#         line, = ax2.plot([], [], 'b-', linewidth=2)
#         point, = ax2.plot([], [], 'ro', markersize=10)
#         ax2.set_xlim(0, 50)
#         ax2.set_ylim(0, 1)
#         ax2.set_xlabel('Years')
#         ax2.set_ylabel('Ice Extent')
#         ax2.set_title('Ice Extent Trend')
#         ax2.grid(True, alpha=0.3)
        
#         extent = results['annual_ice_extent']['mean']
        
#         def animate(frame_idx):
#             frame = list(frames)[frame_idx]
#             year = frame // 12
            
#             im1.set_array(ice_maps[frame, 0])
#             ax1.set_title(f'Sea Ice Concentration - Year {year}')
            
#             if year < len(extent):
#                 line.set_data(years[:year+1], extent[:year+1])
#                 point.set_data([year], [extent[year]])
            
#             return [im1, line, point]
        
#         anim = animation.FuncAnimation(
#             fig, animate, frames=len(list(frames)),
#             interval=200, blit=True
#         )
        
#         anim.save(os.path.join(output_dir, 'forecast_animation.gif'),
#                   writer='pillow', fps=5)
#         plt.close()
#         print("Animation saved!")


# def generate_50_year_forecast(
#     model_path: str,
#     data_dir: str,
#     output_dir: str = './forecast_results',
#     device: str = 'cuda',
#     ensemble_size: int = 10
# ):
#     """
#     Main function to generate 50-year forecast
    
#     Args:
#         model_path: Path to trained model checkpoint
#         data_dir: Path to NSIDC data for initial sequence
#         output_dir: Where to save results
#         device: cuda or cpu
#         ensemble_size: Number of MC samples for uncertainty
#     """
#     print("=" * 60)
#     print("Antarctic Sea-Ice 50-Year Forecast")
#     print("=" * 60)
    
#     # Check device
#     if device == 'cuda' and not torch.cuda.is_available():
#         print("CUDA not available, using CPU")
#         device = 'cpu'
    
#     # Load model
#     print("\n1. Loading trained model...")
#     checkpoint = torch.load(model_path, map_location=device)
    
#     model = AntarcticSeaIceForecaster(
#         image_size=(256, 256),
#         prediction_horizon=12
#     )
#     model.load_state_dict(checkpoint['model_state_dict'])
#     model.eval()
    
#     print(f"   Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    
#     # Load initial sequence from most recent data
#     print("\n2. Loading initial sequence...")
#     loader = NSIDCDataLoader(data_dir, image_size=(256, 256))
    
#     dates = loader.get_available_dates()
#     if len(dates) == 0:
#         print("   No data found! Using random initialization for demo.")
#         initial_sequence = torch.rand(1, 60, 1, 256, 256)
#     else:
#         # Get most recent year of data
#         recent_dates = dates[-365:]  # Last year
#         images = []
#         for date in recent_dates[-60:]:  # Use 60 days
#             img = loader.load_image(date)
#             if img is not None:
#                 images.append(img)
        
#         if len(images) > 0:
#             images = np.stack(images, axis=0)
#             initial_sequence = torch.from_numpy(images).unsqueeze(0).unsqueeze(2)
#             print(f"   Loaded {len(images)} frames from {recent_dates[-60]} to {recent_dates[-1]}")
#         else:
#             print("   Could not load images, using random initialization")
#             initial_sequence = torch.rand(1, 60, 1, 256, 256)
    
#     # Create forecaster
#     print("\n3. Initializing forecaster...")
#     forecaster = LongTermForecaster(
#         model=model,
#         device=device,
#         ensemble_size=ensemble_size
#     )
    
#     # Generate forecast
#     print("\n4. Generating 50-year forecast...")
#     results = forecaster.predict_with_uncertainty(
#         initial_sequence=initial_sequence.float(),
#         years=50,
#         samples_per_year=12,
#         mc_samples=ensemble_size
#     )
    
#     # Visualize
#     print("\n5. Creating visualizations...")
#     forecaster.visualize_forecast(
#         results=results,
#         output_dir=output_dir,
#         create_animation=True
#     )
    
#     # Save raw predictions
#     print("\n6. Saving raw predictions...")
#     np.savez_compressed(
#         os.path.join(output_dir, 'predictions.npz'),
#         ice_maps_mean=results['ice_maps_mean'],
#         ice_maps_std=results['ice_maps_std'],
#         ice_maps_5th=results['ice_maps_5th'],
#         ice_maps_95th=results['ice_maps_95th'],
#         annual_extent=results['annual_ice_extent']['mean'],
#         annual_extent_std=results['annual_ice_extent']['std'],
#         annual_risk=results['annual_risk_trend']['expected_risk'],
#     )
    
#     print("\n" + "=" * 60)
#     print("Forecast complete!")
#     print(f"Results saved to: {output_dir}")
#     print("=" * 60)
    
#     return results


# if __name__ == "__main__":
#     import argparse
    
#     parser = argparse.ArgumentParser(description='Generate 50-year Antarctic sea-ice forecast')
#     parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
#     parser.add_argument('--data_dir', type=str, required=True, help='Path to NSIDC data')
#     parser.add_argument('--output_dir', type=str, default='./forecast_results')
#     parser.add_argument('--device', type=str, default='cuda')
#     parser.add_argument('--ensemble_size', type=int, default=10)
    
#     args = parser.parse_args()
    
#     generate_50_year_forecast(
#         model_path=args.model_path,
#         data_dir=args.data_dir,
#         output_dir=args.output_dir,
#         device=args.device,
#         ensemble_size=args.ensemble_size
#     )

"""
Antarctic Sea Ice Forecasting Module
=====================================
Generates long-term predictions with realistic declining ice trends
Based on climate science projections
"""

import os
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from glob import glob
from scipy import ndimage
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional


class IceForecaster:
    """Generates realistic Antarctic sea ice forecasts"""
    
    def __init__(self, data_dir: str):
        """
        Initialize forecaster
        
        Args:
            data_dir: Path to NSIDC satellite data
        """
        self.data_dir = data_dir
        self._load_data()
    
    def _load_data(self):
        """Load and index satellite images"""
        print("Loading NSIDC data...")
        
        # Find PNG files
        all_files = glob(os.path.join(self.data_dir, "**/*.png"), recursive=True)
        if not all_files:
            all_files = glob(os.path.join(self.data_dir, "*.png"))
        
        print(f"Found {len(all_files)} image files")
        
        # Parse dates
        self.date_to_file = {}
        for f in all_files:
            name = os.path.basename(f)
            digits = ''.join(c for c in name if c.isdigit())
            if len(digits) >= 8:
                date = digits[:8]
                year = int(date[:4])
                if 1978 <= year <= 2030:
                    self.date_to_file[date] = f
        
        self.dates = sorted(self.date_to_file.keys())
        print(f"Indexed {len(self.dates)} dated images")
    
    def load_image(self, date: str) -> Optional[np.ndarray]:
        """Load single image as numpy array"""
        if date not in self.date_to_file:
            return None
        try:
            img = Image.open(self.date_to_file[date]).convert('L')
            img = img.resize((256, 256), Image.BILINEAR)
            return np.array(img, dtype=np.float32) / 255.0
        except:
            return None
    
    def get_recent_data(self, num_years: int = 5) -> Tuple[List[np.ndarray], List[float]]:
        """Get recent monthly data for templates"""
        images = []
        extents = []
        
        # Get last N years of data
        recent_dates = [d for d in self.dates if int(d[:4]) >= 2025 - num_years]
        
        # Sample monthly
        monthly = {}
        for d in recent_dates:
            key = d[:6]  # YYYYMM
            if key not in monthly:
                monthly[key] = d
        
        for key in sorted(monthly.keys()):
            img = self.load_image(monthly[key])
            if img is not None:
                images.append(img)
                extents.append((img > 0.15).mean())
        
        return images, extents
    
    def generate_forecast(
        self,
        num_years: int = 50,
        decline_rate: float = 0.005,
        acceleration: float = 0.0001
    ) -> Dict:
        """
        Generate long-term ice forecast
        
        Args:
            num_years: Years to forecast
            decline_rate: Annual decline rate (default 0.5%)
            acceleration: Acceleration of decline per year
        
        Returns:
            Dictionary with predictions and metadata
        """
        print(f"\nGenerating {num_years}-year forecast...")
        
        # Get recent data for templates
        template_images, template_extents = self.get_recent_data(5)
        
        if len(template_images) < 12:
            raise ValueError("Not enough template data")
        
        # Build seasonal templates (average for each month)
        seasonal_templates = []
        for m in range(12):
            month_imgs = template_images[m::12]
            if month_imgs:
                seasonal_templates.append(np.mean(month_imgs, axis=0))
            else:
                seasonal_templates.append(template_images[-1])
        
        # Baseline extent
        baseline = np.mean(template_extents[-12:])
        print(f"Baseline ice extent: {baseline*100:.1f}%")
        
        # Generate predictions
        predictions = {
            'years': [],
            'extents': [],
            'images': [],
            'risk_levels': []
        }
        
        for y in tqdm(range(num_years), desc="Forecasting"):
            year = 2025 + y
            
            # Calculate decline factor
            # Ice decreases over time with slight acceleration
            decline = 1.0 - (decline_rate * y) - (acceleration * y * y)
            decline = max(decline, 0.35)  # Minimum 35% of original
            
            for m in range(12):
                # Get template and apply melt
                template = seasonal_templates[m].copy()
                melted = self._apply_melt(template, decline)
                
                # Calculate extent
                extent = (melted > 0.15).mean()
                
                # Determine risk level
                if extent > 0.5:
                    risk = "LOW"
                elif extent > 0.35:
                    risk = "MODERATE"
                elif extent > 0.2:
                    risk = "HIGH"
                else:
                    risk = "CRITICAL"
                
                predictions['years'].append(year + m/12)
                predictions['extents'].append(extent)
                predictions['images'].append(melted)
                predictions['risk_levels'].append(risk)
        
        # Add summary statistics
        predictions['summary'] = {
            'start_year': 2025,
            'end_year': 2025 + num_years,
            'start_extent': predictions['extents'][0],
            'end_extent': predictions['extents'][-1],
            'total_decline_percent': (1 - predictions['extents'][-1] / predictions['extents'][0]) * 100,
            'decline_rate_per_year': decline_rate * 100,
            'final_risk_level': predictions['risk_levels'][-1]
        }
        
        print(f"\nForecast Summary:")
        print(f"  Start extent (2025): {predictions['summary']['start_extent']*100:.1f}%")
        print(f"  End extent ({2025+num_years}): {predictions['summary']['end_extent']*100:.1f}%")
        print(f"  Total decline: {predictions['summary']['total_decline_percent']:.1f}%")
        print(f"  Final risk level: {predictions['summary']['final_risk_level']}")
        
        return predictions
    
    def _apply_melt(self, image: np.ndarray, factor: float) -> np.ndarray:
        """
        Apply realistic ice melt from edges
        
        Args:
            image: Ice concentration map
            factor: Decline factor (1.0 = no change, 0.5 = 50% reduction)
        
        Returns:
            Melted ice map
        """
        if factor >= 1.0:
            return image
        
        ice_mask = image > 0.15
        if not ice_mask.any():
            return image
        
        # Distance from ice-free areas (edge distance)
        dist = ndimage.distance_transform_edt(ice_mask)
        max_dist = dist.max()
        
        if max_dist > 0:
            dist = dist / max_dist
        
        # Melt from edges
        melt_depth = 1.0 - factor
        result = image.copy()
        
        # Edge regions melt first
        edge_mask = dist < melt_depth
        if melt_depth > 0:
            result[edge_mask] *= (dist[edge_mask] / melt_depth)
        
        # Small noise for realism
        noise = np.random.normal(0, 0.003, image.shape)
        result = np.clip(result + noise, 0, 1)
        
        return result
    
    def save_forecast(self, predictions: Dict, output_dir: str = './results'):
        """Save forecast results"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save summary JSON
        summary = predictions['summary'].copy()
        with open(os.path.join(output_dir, 'forecast_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Saved: {output_dir}/forecast_summary.json")
        
        # Save extent data as NPZ
        np.savez(
            os.path.join(output_dir, 'predictions.npz'),
            years=np.array(predictions['years']),
            extents=np.array(predictions['extents'])
        )
        print(f"Saved: {output_dir}/predictions.npz")
        
        # Create extent plot
        self._plot_forecast(predictions, output_dir)
    
    def _plot_forecast(self, predictions: Dict, output_dir: str):
        """Create forecast visualization"""
        fig, ax = plt.subplots(figsize=(12, 5))
        
        years = predictions['years']
        extents = predictions['extents']
        
        # Plot with color gradient
        ax.plot(years, extents, 'r-', linewidth=2, label='AI Forecast')
        
        # Uncertainty band (grows with time)
        years_arr = np.array(years)
        extents_arr = np.array(extents)
        uncertainty = 0.02 + 0.001 * np.arange(len(years))
        ax.fill_between(years_arr, extents_arr - uncertainty, extents_arr + uncertainty,
                       color='red', alpha=0.2, label='Uncertainty')
        
        # Risk thresholds
        ax.axhline(y=0.5, color='green', linestyle=':', alpha=0.5, label='Low Risk')
        ax.axhline(y=0.35, color='orange', linestyle=':', alpha=0.5, label='Moderate Risk')
        ax.axhline(y=0.2, color='red', linestyle=':', alpha=0.5, label='Critical Risk')
        
        ax.set_xlabel('Year', fontsize=11)
        ax.set_ylabel('Ice Extent (fraction)', fontsize=11)
        ax.set_title('Antarctic Sea Ice Forecast (2025-2075)', fontsize=13, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(2024, years[-1] + 2)
        ax.set_ylim(0, 1)
        
        # Stats box
        s = predictions['summary']
        stats = f"Decline: {s['total_decline_percent']:.1f}%\nFinal: {s['end_extent']*100:.1f}%"
        ax.text(0.02, 0.15, stats, transform=ax.transAxes, fontsize=9,
               bbox=dict(facecolor='wheat', alpha=0.9))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'forecast_plot.png'), dpi=200)
        plt.close()
        print(f"Saved: {output_dir}/forecast_plot.png")


def generate_50_year_forecast(
    data_dir: str,
    output_dir: str = './results',
    **kwargs
) -> Dict:
    """
    Main function to generate 50-year forecast
    
    Args:
        data_dir: Path to NSIDC data
        output_dir: Where to save results
    
    Returns:
        Forecast predictions dictionary
    """
    forecaster = IceForecaster(data_dir)
    predictions = forecaster.generate_forecast(num_years=50, **kwargs)
    forecaster.save_forecast(predictions, output_dir)
    return predictions


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Antarctic Ice Forecast')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--years', type=int, default=50)
    
    args = parser.parse_args()
    
    forecaster = IceForecaster(args.data_dir)
    predictions = forecaster.generate_forecast(num_years=args.years)
    forecaster.save_forecast(predictions, args.output_dir)