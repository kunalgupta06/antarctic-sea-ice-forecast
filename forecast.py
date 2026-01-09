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
Long-term (50-year) Forecasting Script for Antarctic Sea-Ice
Includes:
- Autoregressive multi-decade predictions
- Monte Carlo dropout for uncertainty
- Ensemble forecasting
- Visualization and analysis
"""
import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap

from config import ForecastConfig, DataConfig, ModelConfig
from model import AntarcticSeaIceForecaster
from data_loader import NSIDCDataLoader, YearlyAggregatedDataset


class LongTermForecaster:
    """
    50-year autoregressive forecaster with uncertainty quantification
    """
    
    def __init__(
        self,
        model: AntarcticSeaIceForecaster,
        device: str = 'cuda',
        ensemble_size: int = 10
    ):
        self.model = model.to(device)
        self.device = device
        self.ensemble_size = ensemble_size
        
        # Custom colormap for sea ice
        self.ice_cmap = LinearSegmentedColormap.from_list(
            'ice',
            [(0, '#1a1a2e'), (0.3, '#16537e'), (0.5, '#1e90ff'),
             (0.7, '#87ceeb'), (1, '#ffffff')]
        )
        
        self.risk_cmap = LinearSegmentedColormap.from_list(
            'risk',
            [(0, '#00ff00'), (0.25, '#7fff00'), (0.5, '#ffff00'),
             (0.75, '#ff7f00'), (1, '#ff0000')]
        )
    
    def enable_dropout(self):
        """Enable dropout for MC sampling"""
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout) or isinstance(module, torch.nn.Dropout2d):
                module.train()
    
    def predict_with_uncertainty(
        self,
        initial_sequence: torch.Tensor,
        years: int = 50,
        samples_per_year: int = 12,
        mc_samples: int = 10
    ) -> Dict[str, np.ndarray]:
        """
        Generate predictions with Monte Carlo dropout uncertainty
        
        Args:
            initial_sequence: (1, T, C, H, W) seed sequence
            years: number of years to forecast
            samples_per_year: predictions per year (12 = monthly)
            mc_samples: number of MC dropout samples
        
        Returns:
            Dictionary with predictions, uncertainty, and statistics
        """
        self.model.eval()
        total_steps = years * samples_per_year
        
        all_predictions = []
        
        print(f"Generating {mc_samples} ensemble forecasts for {years} years...")
        
        for sample_idx in tqdm(range(mc_samples), desc="MC Samples"):
            # Enable dropout for this sample
            self.enable_dropout()
            
            predictions = self._autoregressive_forecast(
                initial_sequence.clone(),
                years=years,
                samples_per_year=samples_per_year
            )
            
            all_predictions.append(predictions)
        
        # Stack all predictions
        ice_maps = np.stack([p['ice_maps'] for p in all_predictions], axis=0)
        habitat_risk = np.stack([p['habitat_risk'] for p in all_predictions], axis=0)
        
        # Compute statistics
        results = {
            'ice_maps_mean': ice_maps.mean(axis=0),
            'ice_maps_std': ice_maps.std(axis=0),
            'ice_maps_median': np.median(ice_maps, axis=0),
            'ice_maps_5th': np.percentile(ice_maps, 5, axis=0),
            'ice_maps_95th': np.percentile(ice_maps, 95, axis=0),
            'habitat_risk_mean': habitat_risk.mean(axis=0),
            'habitat_risk_std': habitat_risk.std(axis=0),
            'all_ice_maps': ice_maps,
            'all_habitat_risk': habitat_risk,
            'years': np.arange(years),
            'months': np.arange(total_steps),
        }
        
        # Compute annual statistics
        results['annual_ice_extent'] = self._compute_annual_extent(ice_maps)
        results['annual_risk_trend'] = self._compute_annual_risk(habitat_risk)
        
        return results
    
    @torch.no_grad()
    def _autoregressive_forecast(
        self,
        initial_sequence: torch.Tensor,
        years: int = 50,
        samples_per_year: int = 12
    ) -> Dict[str, np.ndarray]:
        """Single autoregressive forecast run"""
        
        current_input = initial_sequence.to(self.device)
        
        all_ice = []
        all_risk = []
        
        # Calculate how many steps we need
        total_steps = years * samples_per_year
        steps_done = 0
        
        while steps_done < total_steps:
            # Get predictions (model outputs prediction_horizon steps)
            output = self.model(current_input, forecast_steps=min(12, total_steps - steps_done))
            
            # Extract predictions
            ice_maps = output['ice_maps'].cpu().numpy()
            risk_class = output['habitat_risk_class'].cpu().numpy()
            
            all_ice.append(ice_maps[0])  # Remove batch dim
            all_risk.append(risk_class[0])
            
            steps_done += ice_maps.shape[1]
            
            # Update input sequence with predictions
            predicted = output['ice_maps']  # (1, T_pred, C, H, W)
            
            # Slide window: drop oldest, append newest
            pred_len = predicted.shape[1]
            if current_input.shape[1] > pred_len:
                current_input = torch.cat([
                    current_input[:, pred_len:],
                    predicted
                ], dim=1)
            else:
                current_input = predicted
        
        return {
            'ice_maps': np.concatenate(all_ice, axis=0)[:total_steps],
            'habitat_risk': np.concatenate(all_risk, axis=0)[:total_steps]
        }
    
    def _compute_annual_extent(self, ice_maps: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute annual sea ice extent statistics
        
        Args:
            ice_maps: (ensemble, time, C, H, W) predictions
        """
        ensemble, time, C, H, W = ice_maps.shape
        samples_per_year = 12
        years = time // samples_per_year
        
        # Reshape to yearly
        yearly = ice_maps.reshape(ensemble, years, samples_per_year, C, H, W)
        
        # Sea ice extent: area where concentration > 15%
        extent_threshold = 0.15
        extent = (yearly > extent_threshold).sum(axis=(-3, -2, -1)) / (C * H * W)
        
        # Annual statistics
        annual_mean = extent.mean(axis=2)  # Average over months
        annual_min = extent.min(axis=2)    # Minimum (summer)
        annual_max = extent.max(axis=2)    # Maximum (winter)
        
        return {
            'mean': annual_mean.mean(axis=0),  # Ensemble mean
            'std': annual_mean.std(axis=0),
            'min_extent': annual_min.mean(axis=0),
            'max_extent': annual_max.mean(axis=0),
            '5th_percentile': np.percentile(annual_mean, 5, axis=0),
            '95th_percentile': np.percentile(annual_mean, 95, axis=0),
        }
    
    def _compute_annual_risk(self, habitat_risk: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute annual habitat risk trends"""
        ensemble, time, num_classes = habitat_risk.shape
        samples_per_year = 12
        years = time // samples_per_year
        
        # Reshape to yearly
        yearly = habitat_risk.reshape(ensemble, years, samples_per_year, num_classes)
        
        # Average risk class probability over months
        annual_risk = yearly.mean(axis=2)
        
        # Compute expected risk level (0=minimal, 4=critical)
        risk_levels = np.arange(num_classes)
        expected_risk = (annual_risk * risk_levels).sum(axis=-1)
        
        return {
            'expected_risk': expected_risk.mean(axis=0),
            'risk_std': expected_risk.std(axis=0),
            'class_probabilities': annual_risk.mean(axis=0),
        }
    
    def visualize_forecast(
        self,
        results: Dict[str, np.ndarray],
        output_dir: str,
        create_animation: bool = True
    ):
        """Create comprehensive visualizations"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Annual ice extent trend
        fig, ax = plt.subplots(figsize=(14, 6))
        
        years = np.arange(len(results['annual_ice_extent']['mean']))
        mean = results['annual_ice_extent']['mean']
        std = results['annual_ice_extent']['std']
        p5 = results['annual_ice_extent']['5th_percentile']
        p95 = results['annual_ice_extent']['95th_percentile']
        
        ax.fill_between(years, p5, p95, alpha=0.3, color='blue', label='90% CI')
        ax.fill_between(years, mean - std, mean + std, alpha=0.5, color='blue', label='±1 std')
        ax.plot(years, mean, 'b-', linewidth=2, label='Mean prediction')
        ax.plot(years, results['annual_ice_extent']['min_extent'], 
                'r--', linewidth=1, label='Summer minimum')
        ax.plot(years, results['annual_ice_extent']['max_extent'], 
                'g--', linewidth=1, label='Winter maximum')
        
        ax.set_xlabel('Years from now', fontsize=12)
        ax.set_ylabel('Sea Ice Extent (fraction)', fontsize=12)
        ax.set_title('Antarctic Sea Ice Extent: 50-Year Forecast', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'ice_extent_trend.png'), dpi=300)
        plt.close()
        
        # 2. Habitat risk trend
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        risk = results['annual_risk_trend']
        
        ax1.fill_between(years, 
                         risk['expected_risk'] - risk['risk_std'],
                         risk['expected_risk'] + risk['risk_std'],
                         alpha=0.3, color='red')
        ax1.plot(years, risk['expected_risk'], 'r-', linewidth=2)
        ax1.set_xlabel('Years from now')
        ax1.set_ylabel('Expected Risk Level (0=minimal, 4=critical)')
        ax1.set_title('Penguin Habitat Risk Trajectory')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 4)
        
        # Risk class probabilities over time
        risk_classes = ['Minimal', 'Low', 'Moderate', 'High', 'Critical']
        probs = risk['class_probabilities']
        
        ax2.stackplot(years, probs.T, labels=risk_classes,
                      colors=['green', 'lightgreen', 'yellow', 'orange', 'red'],
                      alpha=0.8)
        ax2.set_xlabel('Years from now')
        ax2.set_ylabel('Probability')
        ax2.set_title('Habitat Risk Class Distribution')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'habitat_risk_trend.png'), dpi=300)
        plt.close()
        
        # 3. Uncertainty over time
        fig, ax = plt.subplots(figsize=(12, 5))
        
        # Compute spatial mean uncertainty over time
        uncertainty = results['ice_maps_std'].mean(axis=(1, 2, 3))
        
        ax.plot(np.arange(len(uncertainty)), uncertainty, 'purple', linewidth=2)
        ax.fill_between(np.arange(len(uncertainty)), 0, uncertainty, alpha=0.3, color='purple')
        ax.set_xlabel('Months from now')
        ax.set_ylabel('Prediction Uncertainty (std)')
        ax.set_title('Forecast Uncertainty Over Time')
        ax.grid(True, alpha=0.3)
        
        # Add year markers
        for year in range(0, 51, 10):
            ax.axvline(year * 12, color='gray', linestyle='--', alpha=0.5)
            ax.text(year * 12, ax.get_ylim()[1], f'Year {year}', 
                   ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'uncertainty_trend.png'), dpi=300)
        plt.close()
        
        # 4. Sample ice maps at key timepoints
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        
        timepoints = [0, 60, 120, 240, 360, 480, 540, 570, 594, 599]  # Months
        timepoints = [t for t in timepoints if t < len(results['ice_maps_mean'])]
        
        for idx, t in enumerate(timepoints[:10]):
            row = idx // 5
            col = idx % 5
            
            ax = axes[row, col]
            
            mean_map = results['ice_maps_mean'][t, 0]
            uncertainty_map = results['ice_maps_std'][t, 0]
            
            # Overlay: mean with uncertainty as alpha
            im = ax.imshow(mean_map, cmap=self.ice_cmap, vmin=0, vmax=1)
            ax.contour(uncertainty_map, levels=[0.1, 0.2, 0.3], 
                      colors='red', linewidths=0.5, alpha=0.5)
            
            year = t // 12
            month = t % 12 + 1
            ax.set_title(f'Year {year}, Month {month}')
            ax.axis('off')
        
        plt.suptitle('Predicted Sea Ice Concentration Over 50 Years', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'ice_map_samples.png'), dpi=300)
        plt.close()
        
        # 5. Create animation if requested
        if create_animation:
            self._create_animation(results, output_dir)
        
        print(f"Visualizations saved to {output_dir}")
        
        # 6. Save numerical summary
        summary = {
            'years_forecasted': 50,
            'initial_ice_extent': float(results['annual_ice_extent']['mean'][0]),
            'final_ice_extent': float(results['annual_ice_extent']['mean'][-1]),
            'extent_change_percent': float(
                (results['annual_ice_extent']['mean'][-1] - results['annual_ice_extent']['mean'][0]) 
                / results['annual_ice_extent']['mean'][0] * 100
            ),
            'initial_risk_level': float(results['annual_risk_trend']['expected_risk'][0]),
            'final_risk_level': float(results['annual_risk_trend']['expected_risk'][-1]),
            'avg_uncertainty_first_decade': float(results['ice_maps_std'][:120].mean()),
            'avg_uncertainty_last_decade': float(results['ice_maps_std'][-120:].mean()),
        }
        
        with open(os.path.join(output_dir, 'forecast_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("\n📊 Forecast Summary:")
        print(f"  Ice Extent Change: {summary['extent_change_percent']:.1f}%")
        print(f"  Risk Level: {summary['initial_risk_level']:.2f} → {summary['final_risk_level']:.2f}")
        print(f"  Uncertainty Growth: {summary['avg_uncertainty_first_decade']:.3f} → {summary['avg_uncertainty_last_decade']:.3f}")
    
    def _create_animation(self, results: Dict[str, np.ndarray], output_dir: str):
        """Create animated visualization"""
        print("Creating animation...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        ice_maps = results['ice_maps_mean']
        
        # Sample every 6 months for animation speed
        frames = range(0, len(ice_maps), 6)
        
        im1 = ax1.imshow(ice_maps[0, 0], cmap=self.ice_cmap, vmin=0, vmax=1)
        ax1.set_title('Sea Ice Concentration')
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1, label='Concentration')
        
        # Line plot for extent
        years = np.arange(len(results['annual_ice_extent']['mean']))
        line, = ax2.plot([], [], 'b-', linewidth=2)
        point, = ax2.plot([], [], 'ro', markersize=10)
        ax2.set_xlim(0, 50)
        ax2.set_ylim(0, 1)
        ax2.set_xlabel('Years')
        ax2.set_ylabel('Ice Extent')
        ax2.set_title('Ice Extent Trend')
        ax2.grid(True, alpha=0.3)
        
        extent = results['annual_ice_extent']['mean']
        
        def animate(frame_idx):
            frame = list(frames)[frame_idx]
            year = frame // 12
            
            im1.set_array(ice_maps[frame, 0])
            ax1.set_title(f'Sea Ice Concentration - Year {year}')
            
            if year < len(extent):
                line.set_data(years[:year+1], extent[:year+1])
                point.set_data([year], [extent[year]])
            
            return [im1, line, point]
        
        anim = animation.FuncAnimation(
            fig, animate, frames=len(list(frames)),
            interval=200, blit=True
        )
        
        anim.save(os.path.join(output_dir, 'forecast_animation.gif'),
                  writer='pillow', fps=5)
        plt.close()
        print("Animation saved!")


def generate_50_year_forecast(
    model_path: str,
    data_dir: str,
    output_dir: str = './forecast_results',
    device: str = 'cuda',
    ensemble_size: int = 10,
    forecast_years: int = 50
):
    """
    Main function to generate long-term forecast
    
    Args:
        model_path: Path to trained model checkpoint
        data_dir: Path to NSIDC data for initial sequence
        output_dir: Where to save results
        device: cuda or cpu
        ensemble_size: Number of MC samples for uncertainty
        forecast_years: Number of years to forecast (default 50)
    """
    print("=" * 60)
    print(f"Antarctic Sea-Ice {forecast_years}-Year Forecast")
    print("=" * 60)
    
    # Check device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    
    # Load model
    print("\n1. Loading trained model...")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Use the SAME small model architecture as training (for CPU)
    # This must match what was used in train.py for CPU training
    model = AntarcticSeaIceForecaster(
        image_size=(64, 64),
        cnn_channels=(16, 32, 64),
        convlstm_hidden=32,
        convlstm_layers=1,
        transformer_dim=64,
        transformer_heads=4,
        transformer_layers=2,
        prediction_horizon=7
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"   Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Load initial sequence from most recent data
    print("\n2. Loading initial sequence...")
    loader = NSIDCDataLoader(data_dir, image_size=(64, 64))  # Match model size
    
    dates = loader.get_available_dates()
    if len(dates) == 0:
        print("   No data found! Using random initialization for demo.")
        initial_sequence = torch.rand(1, 30, 1, 64, 64)
    else:
        # Get most recent data
        recent_dates = dates[-365:]  # Last year
        images = []
        for date in recent_dates[-30:]:  # Use 30 days
            img = loader.load_image(date)
            if img is not None:
                images.append(img)
        
        if len(images) > 0:
            images = np.stack(images, axis=0)
            initial_sequence = torch.from_numpy(images).unsqueeze(0).unsqueeze(2)
            print(f"   Loaded {len(images)} frames from {recent_dates[-30]} to {recent_dates[-1]}")
        else:
            print("   Could not load images, using random initialization")
            initial_sequence = torch.rand(1, 30, 1, 64, 64)
    
    # Create forecaster
    print("\n3. Initializing forecaster...")
    forecaster = LongTermForecaster(
        model=model,
        device=device,
        ensemble_size=min(ensemble_size, 3)  # Limit for CPU speed
    )
    
    # Generate forecast
    print(f"\n4. Generating {forecast_years}-year forecast...")
    print(f"   ⏱️  Estimated time: ~{forecast_years * 2} minutes on CPU...")
    
    results = forecaster.predict_with_uncertainty(
        initial_sequence=initial_sequence.float(),
        years=forecast_years,
        samples_per_year=12,
        mc_samples=min(ensemble_size, 3)  # Limit for CPU
    )
    
    # Visualize
    print("\n5. Creating visualizations...")
    forecaster.visualize_forecast(
        results=results,
        output_dir=output_dir,
        create_animation=True
    )
    
    # Save raw predictions
    print("\n6. Saving raw predictions...")
    np.savez_compressed(
        os.path.join(output_dir, 'predictions.npz'),
        ice_maps_mean=results['ice_maps_mean'],
        ice_maps_std=results['ice_maps_std'],
        ice_maps_5th=results['ice_maps_5th'],
        ice_maps_95th=results['ice_maps_95th'],
        annual_extent=results['annual_ice_extent']['mean'],
        annual_extent_std=results['annual_ice_extent']['std'],
        annual_risk=results['annual_risk_trend']['expected_risk'],
    )
    
    print("\n" + "=" * 60)
    print("Forecast complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate 50-year Antarctic sea-ice forecast')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to NSIDC data')
    parser.add_argument('--output_dir', type=str, default='./forecast_results')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--ensemble_size', type=int, default=10)
    
    args = parser.parse_args()
    
    generate_50_year_forecast(
        model_path=args.model_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=args.device,
        ensemble_size=args.ensemble_size
    )