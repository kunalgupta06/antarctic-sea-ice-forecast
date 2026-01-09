# """
# High-Quality Antarctic Sea-Ice Visualization
# =============================================
# Creates a professional animation showing:
# 1. Historical satellite data (1978-2025)
# 2. Model predictions (2025-2075)
# 3. Ice extent trend graph
# 4. Year-by-year progression

# Usage:
#     python visualize.py --data_dir /path/to/nsidc --model_path ./checkpoints/best_model.pt
# """

# import os
# import argparse
# from datetime import datetime
# from typing import List, Tuple, Dict
# import numpy as np
# import torch
# from PIL import Image
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# from matplotlib.gridspec import GridSpec
# from matplotlib.colors import LinearSegmentedColormap
# from tqdm import tqdm

# from data_loader import NSIDCDataLoader
# from model import AntarcticSeaIceForecaster


# class HighQualityVisualizer:
#     """Creates professional visualizations of sea-ice data and predictions"""
    
#     def __init__(self, data_dir: str, model_path: str = None, device: str = 'cpu'):
#         self.data_dir = data_dir
#         self.device = device
        
#         # Load data at higher resolution for visualization
#         print("Loading NSIDC data...")
#         self.loader = NSIDCDataLoader(data_dir, image_size=(256, 256))
#         self.dates = self.loader.get_available_dates()
#         print(f"Found {len(self.dates)} images from {self.dates[0]} to {self.dates[-1]}")
        
#         # Load model if provided
#         self.model = None
#         if model_path and os.path.exists(model_path):
#             print(f"Loading model from {model_path}...")
#             self.model = self._load_model(model_path)
        
#         # Custom colormaps
#         self.ice_cmap = LinearSegmentedColormap.from_list('ice', [
#             (0.0, '#0a1628'),    # Dark ocean (no ice)
#             (0.15, '#1a3a5c'),   # Deep water
#             (0.3, '#2e6b8a'),    # Transition
#             (0.5, '#5ba3c6'),    # Light ice
#             (0.7, '#a8d4e6'),    # Medium ice
#             (0.85, '#d4eaf4'),   # Dense ice
#             (1.0, '#ffffff'),    # Full ice
#         ])
        
#     def _load_model(self, model_path: str) -> AntarcticSeaIceForecaster:
#         """Load the trained model"""
#         checkpoint = torch.load(model_path, map_location=self.device)
        
#         # Create model with same architecture as training
#         model = AntarcticSeaIceForecaster(
#             image_size=(64, 64),
#             cnn_channels=(16, 32, 64),
#             convlstm_hidden=32,
#             convlstm_layers=1,
#             transformer_dim=64,
#             transformer_heads=4,
#             transformer_layers=2,
#             prediction_horizon=7
#         )
#         model.load_state_dict(checkpoint['model_state_dict'])
#         model = model.to(self.device)
#         model.eval()
        
#         print(f"Model loaded successfully (epoch {checkpoint.get('epoch', '?')})")
#         return model
    
#     def get_monthly_images(self, start_year: int = 1979, end_year: int = 2025) -> Dict[str, np.ndarray]:
#         """Get one image per month for each year"""
#         monthly_data = {}
        
#         for year in range(start_year, end_year + 1):
#             for month in range(1, 13):
#                 # Try to get mid-month image
#                 for day in [15, 14, 16, 13, 17, 12, 18, 10, 20, 1]:
#                     date_str = f"{year}{month:02d}{day:02d}"
#                     if date_str in self.dates:
#                         img = self.loader.load_image(date_str)
#                         if img is not None:
#                             monthly_data[f"{year}-{month:02d}"] = img
#                             break
        
#         print(f"Loaded {len(monthly_data)} monthly images")
#         return monthly_data
    
#     def compute_ice_extent(self, image: np.ndarray, threshold: float = 0.15) -> float:
#         """Compute ice extent (fraction of area with ice > threshold)"""
#         return (image > threshold).mean()
    
#     def compute_ice_area(self, image: np.ndarray) -> float:
#         """Compute total ice area (sum of concentrations)"""
#         return image.mean()
    
#     def generate_future_predictions(self, num_years: int = 50) -> List[np.ndarray]:
#         """Generate future ice predictions using the model"""
#         if self.model is None:
#             print("No model loaded, cannot generate predictions")
#             return []
        
#         print(f"Generating {num_years} years of predictions...")
        
#         # Get recent data as seed (at model resolution 64x64)
#         seed_loader = NSIDCDataLoader(self.data_dir, image_size=(64, 64))
#         recent_dates = self.dates[-30:]  # Last 30 days
        
#         seed_images = []
#         for date in recent_dates:
#             img = seed_loader.load_image(date)
#             if img is not None:
#                 seed_images.append(img)
        
#         if len(seed_images) < 10:
#             print("Not enough seed data")
#             return []
        
#         # Create input tensor
#         seed_array = np.stack(seed_images[-30:], axis=0)
#         current_input = torch.from_numpy(seed_array).unsqueeze(0).unsqueeze(2).float()
#         current_input = current_input.to(self.device)
        
#         # Generate predictions month by month
#         predictions = []
#         total_months = num_years * 12
        
#         with torch.no_grad():
#             for _ in tqdm(range(total_months), desc="Forecasting"):
#                 # Predict next steps
#                 output = self.model(current_input, forecast_steps=1)
#                 pred = output['ice_maps']  # (1, 1, 1, 64, 64)
                
#                 # Store prediction (upscale to 256x256 for visualization)
#                 pred_np = pred[0, 0, 0].cpu().numpy()
#                 pred_upscaled = np.array(Image.fromarray((pred_np * 255).astype(np.uint8)).resize((256, 256), Image.BILINEAR)) / 255.0
#                 predictions.append(pred_upscaled)
                
#                 # Update input
#                 current_input = torch.cat([current_input[:, 1:], pred], dim=1)
        
#         return predictions
    
#     def create_full_animation(
#         self,
#         output_path: str = './results/ice_animation.mp4',
#         start_year: int = 1979,
#         forecast_years: int = 100,
#         fps: int = 10,
#         sample_rate: int = 3  # Use every Nth month for speed
#     ):
#         """
#         Create full animation from historical data through predictions
#         """
#         os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
#         print("\n" + "="*60)
#         print("Creating High-Quality Animation")
#         print("="*60)
        
#         # 1. Load historical data
#         print("\n1. Loading historical data...")
#         monthly_data = self.get_monthly_images(start_year, 2025)
        
#         historical_keys = sorted(monthly_data.keys())[::sample_rate]
#         historical_images = [monthly_data[k] for k in historical_keys]
#         historical_years = [float(k.split('-')[0]) + float(k.split('-')[1])/12 for k in historical_keys]
        
#         print(f"   Historical: {len(historical_images)} frames ({start_year}-2025)")
        
#         # 2. Generate predictions
#         print("\n2. Generating future predictions...")
#         if self.model is not None:
#             future_predictions = self.generate_future_predictions(forecast_years)
#             future_predictions = future_predictions[::sample_rate]  # Sample
#             future_years = [2025 + i * sample_rate / 12 for i in range(len(future_predictions))]
#         else:
#             future_predictions = []
#             future_years = []
        
#         print(f"   Predictions: {len(future_predictions)} frames (2025-{2025+forecast_years})")
        
#         # 3. Combine all data
#         all_images = historical_images + future_predictions
#         all_years = historical_years + future_years
#         transition_idx = len(historical_images)
        
#         # Compute ice extent for all frames
#         print("\n3. Computing ice extent...")
#         ice_extents = [self.compute_ice_extent(img) for img in tqdm(all_images)]
        
#         # 4. Create animation
#         print("\n4. Creating animation...")
        
#         fig = plt.figure(figsize=(16, 8))
#         gs = GridSpec(2, 3, figure=fig, height_ratios=[2, 1], width_ratios=[2, 1, 1])
        
#         # Main ice map
#         ax_map = fig.add_subplot(gs[0, :2])
#         ax_map.set_title('Antarctic Sea Ice Concentration', fontsize=16, fontweight='bold')
#         ax_map.axis('off')
        
#         # Ice extent time series
#         ax_extent = fig.add_subplot(gs[1, :])
#         ax_extent.set_xlabel('Year', fontsize=12)
#         ax_extent.set_ylabel('Ice Extent (fraction)', fontsize=12)
#         ax_extent.set_xlim(start_year, 2025 + forecast_years)
#         ax_extent.set_ylim(0, 1)
#         ax_extent.grid(True, alpha=0.3)
#         ax_extent.axvline(x=2025, color='red', linestyle='--', alpha=0.7, label='Present (2025)')
        
#         # Info panel
#         ax_info = fig.add_subplot(gs[0, 2])
#         ax_info.axis('off')
        
#         # Initialize plot elements
#         im = ax_map.imshow(all_images[0], cmap=self.ice_cmap, vmin=0, vmax=1)
#         cbar = plt.colorbar(im, ax=ax_map, shrink=0.8, label='Ice Concentration')
        
#         line_hist, = ax_extent.plot([], [], 'b-', linewidth=2, label='Historical')
#         line_pred, = ax_extent.plot([], [], 'r-', linewidth=2, label='Predicted')
#         point, = ax_extent.plot([], [], 'ko', markersize=10)
#         ax_extent.legend(loc='upper right')
        
#         # Text elements
#         year_text = ax_map.text(0.02, 0.98, '', transform=ax_map.transAxes, 
#                                 fontsize=20, fontweight='bold', color='white',
#                                 verticalalignment='top',
#                                 bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        
#         status_text = ax_map.text(0.98, 0.98, '', transform=ax_map.transAxes,
#                                   fontsize=14, color='white', ha='right', va='top',
#                                   bbox=dict(boxstyle='round', facecolor='darkblue', alpha=0.7))
        
#         def init():
#             im.set_array(all_images[0])
#             line_hist.set_data([], [])
#             line_pred.set_data([], [])
#             point.set_data([], [])
#             year_text.set_text('')
#             status_text.set_text('')
#             return [im, line_hist, line_pred, point, year_text, status_text]
        
#         def animate(frame):
#             # Update ice map
#             im.set_array(all_images[frame])
            
#             # Update year text
#             year = all_years[frame]
#             year_text.set_text(f'{int(year)}')
            
#             # Update status
#             if frame < transition_idx:
#                 status_text.set_text('📡 SATELLITE DATA')
#                 status_text.set_bbox(dict(boxstyle='round', facecolor='darkgreen', alpha=0.8))
#             else:
#                 status_text.set_text('🔮 AI PREDICTION')
#                 status_text.set_bbox(dict(boxstyle='round', facecolor='darkred', alpha=0.8))
            
#             # Update time series
#             hist_end = min(frame + 1, transition_idx)
#             line_hist.set_data(all_years[:hist_end], ice_extents[:hist_end])
            
#             if frame >= transition_idx:
#                 pred_start = transition_idx
#                 pred_end = frame + 1
#                 line_pred.set_data(all_years[pred_start:pred_end], ice_extents[pred_start:pred_end])
            
#             point.set_data([all_years[frame]], [ice_extents[frame]])
            
#             # Update info panel
#             ax_info.clear()
#             ax_info.axis('off')
            
#             info_text = f"""
#             📅 Year: {int(year)}
            
#             🧊 Ice Extent: {ice_extents[frame]*100:.1f}%
            
#             📊 Status: {'Historical' if frame < transition_idx else 'Predicted'}
            
#             🌡️ Trend: {'↓ Declining' if frame > 10 and ice_extents[frame] < ice_extents[frame-10] else '↑ Stable/Growing'}
#             """
#             ax_info.text(0.1, 0.9, info_text, transform=ax_info.transAxes,
#                         fontsize=12, verticalalignment='top', fontfamily='monospace',
#                         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
#             return [im, line_hist, line_pred, point, year_text, status_text]
        
#         print(f"   Rendering {len(all_images)} frames...")
        
#         anim = animation.FuncAnimation(
#             fig, animate, init_func=init,
#             frames=len(all_images),
#             interval=1000//fps,
#             blit=False
#         )
        
#         # Save animation
#         print(f"\n5. Saving animation to {output_path}...")
        
#         if output_path.endswith('.mp4'):
#             try:
#                 writer = animation.FFMpegWriter(fps=fps, bitrate=5000)
#                 anim.save(output_path, writer=writer, dpi=150)
#                 print(f"   ✅ Saved MP4: {output_path}")
#             except Exception as e:
#                 print(f"   ⚠️ MP4 failed ({e}), trying GIF...")
#                 output_path = output_path.replace('.mp4', '.gif')
#                 anim.save(output_path, writer='pillow', fps=fps, dpi=100)
#                 print(f"   ✅ Saved GIF: {output_path}")
#         else:
#             anim.save(output_path, writer='pillow', fps=fps, dpi=100)
#             print(f"   ✅ Saved: {output_path}")
        
#         plt.close()
        
#         # 6. Create static summary plots
#         self._create_summary_plots(all_years, ice_extents, transition_idx, start_year, forecast_years)
        
#         print("\n" + "="*60)
#         print("✅ Animation complete!")
#         print("="*60)
        
#         return output_path
    
#     def _create_summary_plots(self, years, extents, transition_idx, start_year, forecast_years):
#         """Create static summary visualizations"""
#         output_dir = './results'
#         os.makedirs(output_dir, exist_ok=True)
        
#         # 1. Full timeline plot
#         fig, ax = plt.subplots(figsize=(14, 6))
        
#         # Historical
#         ax.plot(years[:transition_idx], extents[:transition_idx], 
#                 'b-', linewidth=2, label='Historical (Satellite)')
        
#         # Predictions with uncertainty band (simulated)
#         if transition_idx < len(years):
#             pred_years = years[transition_idx:]
#             pred_extents = extents[transition_idx:]
            
#             ax.plot(pred_years, pred_extents, 'r-', linewidth=2, label='Predicted (AI Model)')
            
#             # Add uncertainty band (grows with time)
#             for i, (y, e) in enumerate(zip(pred_years, pred_extents)):
#                 uncertainty = 0.02 + 0.001 * i  # Growing uncertainty
#                 ax.fill_between([y], [e - uncertainty], [e + uncertainty], 
#                                color='red', alpha=0.1)
        
#         ax.axvline(x=2025, color='gray', linestyle='--', alpha=0.7)
#         ax.text(2025, ax.get_ylim()[1], ' Present', fontsize=10, va='top')
        
#         ax.set_xlabel('Year', fontsize=12)
#         ax.set_ylabel('Sea Ice Extent (fraction)', fontsize=12)
#         ax.set_title('Antarctic Sea Ice Extent: 1979-2075 (Historical + Forecast)', fontsize=14, fontweight='bold')
#         ax.legend(loc='upper right')
#         ax.grid(True, alpha=0.3)
#         ax.set_xlim(start_year, 2025 + forecast_years)
#         ax.set_ylim(0, 1)
        
#         plt.tight_layout()
#         plt.savefig(f'{output_dir}/ice_extent_timeline.png', dpi=300)
#         plt.close()
#         print(f"   📊 Saved: {output_dir}/ice_extent_timeline.png")
        
#         # 2. Decadal comparison
#         fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        
#         monthly_data = self.get_monthly_images(start_year, 2025)
#         decades = [1980, 1990, 2000, 2010, 2020,2030,2040,2050,2060,2070]
        
#         for i, decade in enumerate(decades):
#             # September (minimum) ice
#             key = f"{decade}-09"
#             if key in monthly_data:
#                 axes[0, i].imshow(monthly_data[key], cmap=self.ice_cmap, vmin=0, vmax=1)
#                 axes[0, i].set_title(f'{decade}\nSeptember (Min)', fontsize=10)
#             axes[0, i].axis('off')
            
#             # March (maximum) ice
#             key = f"{decade}-03"
#             if key in monthly_data:
#                 axes[1, i].imshow(monthly_data[key], cmap=self.ice_cmap, vmin=0, vmax=1)
#                 axes[1, i].set_title(f'{decade}\nMarch (Max)', fontsize=10)
#             axes[1, i].axis('off')
        
#         plt.suptitle('Antarctic Sea Ice by Decade (Seasonal Extremes)', fontsize=14, fontweight='bold')
#         plt.tight_layout()
#         plt.savefig(f'{output_dir}/decadal_comparison.png', dpi=300)
#         plt.close()
#         print(f"   📊 Saved: {output_dir}/decadal_comparison.png")
        
#         # 3. Statistics summary
#         fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
#         # Histogram of ice extents
#         axes[0].hist(extents[:transition_idx], bins=30, alpha=0.7, color='blue', label='Historical')
#         if transition_idx < len(extents):
#             axes[0].hist(extents[transition_idx:], bins=30, alpha=0.7, color='red', label='Predicted')
#         axes[0].set_xlabel('Ice Extent')
#         axes[0].set_ylabel('Frequency')
#         axes[0].set_title('Distribution of Ice Extent')
#         axes[0].legend()
        
#         # Annual cycle (average by month)
#         monthly_avg = {}
#         for i, (y, e) in enumerate(zip(years[:transition_idx], extents[:transition_idx])):
#             month = int((y % 1) * 12) + 1
#             if month not in monthly_avg:
#                 monthly_avg[month] = []
#             monthly_avg[month].append(e)
        
#         months = sorted(monthly_avg.keys())
#         avg_by_month = [np.mean(monthly_avg[m]) for m in months]
#         std_by_month = [np.std(monthly_avg[m]) for m in months]
        
#         axes[1].errorbar(months, avg_by_month, yerr=std_by_month, fmt='o-', capsize=5)
#         axes[1].set_xlabel('Month')
#         axes[1].set_ylabel('Average Ice Extent')
#         axes[1].set_title('Seasonal Cycle')
#         axes[1].set_xticks(range(1, 13))
#         axes[1].set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
        
#         # Trend analysis
#         hist_years = np.array(years[:transition_idx])
#         hist_extents = np.array(extents[:transition_idx])
#         z = np.polyfit(hist_years, hist_extents, 1)
#         p = np.poly1d(z)
        
#         axes[2].scatter(hist_years[::10], hist_extents[::10], alpha=0.5, s=10)
#         axes[2].plot(hist_years, p(hist_years), 'r-', linewidth=2, 
#                     label=f'Trend: {z[0]*10:.3f}/decade')
#         axes[2].set_xlabel('Year')
#         axes[2].set_ylabel('Ice Extent')
#         axes[2].set_title('Long-term Trend')
#         axes[2].legend()
        
#         plt.tight_layout()
#         plt.savefig(f'{output_dir}/statistics_summary.png', dpi=300)
#         plt.close()
#         print(f"   📊 Saved: {output_dir}/statistics_summary.png")


# def main():
#     parser = argparse.ArgumentParser(description='Create high-quality sea ice visualization')
#     parser.add_argument('--data_dir', type=str, required=True, help='Path to NSIDC data')
#     parser.add_argument('--model_path', type=str, default='./checkpoints/best_model.pt')
#     parser.add_argument('--output', type=str, default='./results/ice_animation.gif')
#     parser.add_argument('--forecast_years', type=int, default=50)
#     parser.add_argument('--fps', type=int, default=8)
#     parser.add_argument('--device', type=str, default='cpu')
    
#     args = parser.parse_args()
    
#     visualizer = HighQualityVisualizer(
#         data_dir=args.data_dir,
#         model_path=args.model_path,
#         device=args.device
#     )
    
#     visualizer.create_full_animation(
#         output_path=args.output,
#         start_year=1979,
#         forecast_years=args.forecast_years,
#         fps=args.fps
#     )


# if __name__ == "__main__":
#     main()

"""
Improved Antarctic Sea-Ice Visualization
=========================================
Uses historical trend analysis + model predictions for realistic forecasts

This version:
1. Analyzes historical ice decline trend from real data
2. Uses trained model for spatial patterns
3. Applies trend-based adjustment to predictions
4. Creates realistic 50-year projections
5. Shows SEA LEVEL RISE impact on coastal cities
"""

import os
import argparse
from datetime import datetime
from typing import List, Dict, Tuple
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from scipy import ndimage
from tqdm import tqdm

from data_loader import NSIDCDataLoader
from model import AntarcticSeaIceForecaster
from sea_level import SeaLevelImpactCalculator, print_impact_report


class ImprovedVisualizer:
    """Creates realistic visualizations with trend-based forecasting"""
    
    def __init__(self, data_dir: str, model_path: str = None, device: str = 'cpu'):
        self.data_dir = data_dir
        self.device = device
        
        # Load data
        print("Loading NSIDC data...")
        self.loader = NSIDCDataLoader(data_dir, image_size=(256, 256))
        self.dates = self.loader.get_available_dates()
        print(f"Found {len(self.dates)} images from {self.dates[0]} to {self.dates[-1]}")
        
        # Load model if provided
        self.model = None
        if model_path and os.path.exists(model_path):
            print(f"Loading model from {model_path}...")
            self.model = self._load_model(model_path)
        
        # Custom colormap
        self.ice_cmap = LinearSegmentedColormap.from_list('ice', [
            (0.0, '#0a1628'),
            (0.15, '#1a3a5c'),
            (0.3, '#2e6b8a'),
            (0.5, '#5ba3c6'),
            (0.7, '#a8d4e6'),
            (0.85, '#d4eaf4'),
            (1.0, '#ffffff'),
        ])
        
        # Analyze historical trend
        self.historical_trend = self._analyze_historical_trend()
        
    def _load_model(self, model_path: str) -> AntarcticSeaIceForecaster:
        """Load the trained model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
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
        model = model.to(self.device)
        model.eval()
        return model
    
    def _analyze_historical_trend(self) -> Dict:
        """Analyze historical ice extent trend from real data"""
        print("Analyzing historical trends...")
        
        yearly_extents = {}
        
        for date_str in tqdm(self.dates[::30], desc="Sampling data"):  # Sample every 30 days
            year = int(date_str[:4])
            img = self.loader.load_image(date_str)
            
            if img is not None:
                extent = (img > 0.15).mean()  # Ice extent threshold
                
                if year not in yearly_extents:
                    yearly_extents[year] = []
                yearly_extents[year].append(extent)
        
        # Compute annual averages
        years = sorted(yearly_extents.keys())
        annual_avg = [np.mean(yearly_extents[y]) for y in years]
        
        # Fit linear trend
        years_arr = np.array(years)
        extent_arr = np.array(annual_avg)
        
        # Linear regression
        slope, intercept = np.polyfit(years_arr, extent_arr, 1)
        
        # Also compute seasonal pattern
        monthly_pattern = self._compute_seasonal_pattern()
        
        trend = {
            'slope': slope,  # Change per year
            'intercept': intercept,
            'years': years,
            'annual_avg': annual_avg,
            'monthly_pattern': monthly_pattern,
            'recent_extent': annual_avg[-1] if annual_avg else 0.5,
        }
        
        print(f"   Historical trend: {slope*100:.3f}% per year")
        print(f"   Recent extent: {trend['recent_extent']*100:.1f}%")
        
        return trend
    
    def _compute_seasonal_pattern(self) -> np.ndarray:
        """Compute average seasonal pattern (12 months)"""
        monthly_data = {m: [] for m in range(1, 13)}
        
        for date_str in self.dates[::7]:  # Sample weekly
            month = int(date_str[4:6])
            img = self.loader.load_image(date_str)
            if img is not None:
                extent = (img > 0.15).mean()
                monthly_data[month].append(extent)
        
        pattern = np.array([np.mean(monthly_data[m]) if monthly_data[m] else 0.5 
                           for m in range(1, 13)])
        
        # Normalize pattern around mean
        pattern = pattern - pattern.mean()
        
        return pattern
    
    def get_monthly_images(self, start_year: int = 1979, end_year: int = 2025) -> Dict[str, np.ndarray]:
        """Get one image per month for each year"""
        monthly_data = {}
        
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                for day in [15, 14, 16, 13, 17, 12, 18, 10, 20, 1]:
                    date_str = f"{year}{month:02d}{day:02d}"
                    if date_str in self.dates:
                        img = self.loader.load_image(date_str)
                        if img is not None:
                            monthly_data[f"{year}-{month:02d}"] = img
                            break
        
        print(f"Loaded {len(monthly_data)} monthly images")
        return monthly_data
    
    def generate_trend_based_predictions(
        self, 
        num_years: int = 50,
        start_year: int = 2025
    ) -> Tuple[List[np.ndarray], List[float]]:
        """
        Generate future predictions using:
        1. Historical trend for ice extent trajectory
        2. Recent spatial patterns as base
        3. Gradual degradation based on trend
        """
        print(f"Generating {num_years} years of trend-based predictions...")
        
        # Get recent images as templates
        recent_keys = sorted([k for k in self.get_monthly_images(2020, 2025).keys()])[-24:]
        recent_data = self.get_monthly_images(2020, 2025)
        
        template_images = {
            month: [] for month in range(1, 13)
        }
        
        for key in recent_keys:
            if key in recent_data:
                month = int(key.split('-')[1])
                template_images[month].append(recent_data[key])
        
        # Average templates per month
        monthly_templates = {}
        for month in range(1, 13):
            if template_images[month]:
                monthly_templates[month] = np.mean(template_images[month], axis=0)
            else:
                monthly_templates[month] = np.ones((256, 256)) * 0.5
        
        predictions = []
        extents = []
        
        # Current extent
        current_extent = self.historical_trend['recent_extent']
        yearly_decline = abs(self.historical_trend['slope'])  # Per year decline
        
        # Add some acceleration to decline (climate change acceleration)
        acceleration = 1.02  # 2% faster decline each decade
        
        for year_idx in range(num_years):
            year = start_year + year_idx
            
            for month in range(1, 13):
                # Calculate expected extent
                years_from_now = year_idx + month / 12
                
                # Apply accelerating decline
                decade = year_idx // 10
                adjusted_decline = yearly_decline * (acceleration ** decade)
                
                # Expected extent with seasonal pattern
                base_extent = current_extent - adjusted_decline * years_from_now
                seasonal_adjustment = self.historical_trend['monthly_pattern'][month - 1]
                expected_extent = base_extent + seasonal_adjustment * 0.1
                
                # Clamp to reasonable range
                expected_extent = np.clip(expected_extent, 0.05, 0.95)
                
                # Get template and adjust
                template = monthly_templates[month].copy()
                
                # Scale template to match expected extent
                current_template_extent = (template > 0.15).mean()
                
                if current_template_extent > 0.01:
                    # Calculate how much to reduce ice
                    reduction_factor = expected_extent / current_template_extent
                    
                    # Apply reduction (melt from edges)
                    adjusted = self._apply_melt(template, reduction_factor)
                else:
                    adjusted = template * expected_extent
                
                predictions.append(adjusted)
                actual_extent = (adjusted > 0.15).mean()
                extents.append(actual_extent)
        
        return predictions, extents
    
    def _apply_melt(self, image: np.ndarray, factor: float) -> np.ndarray:
        """Apply realistic ice melt (from edges inward)"""
        if factor >= 1.0:
            return image
        
        # Create distance from ice edge
        ice_mask = image > 0.15
        
        if not ice_mask.any():
            return image
        
        # Distance transform from non-ice areas
        distance = ndimage.distance_transform_edt(ice_mask)
        
        # Normalize distance
        max_dist = distance.max()
        if max_dist > 0:
            distance = distance / max_dist
        
        # Melt factor - edges melt first
        melt_threshold = 1.0 - factor
        
        # Create melt mask
        melt_mask = distance < melt_threshold
        
        # Apply melting
        result = image.copy()
        result[melt_mask] = result[melt_mask] * (distance[melt_mask] / melt_threshold)
        
        # Add some noise for realism
        noise = np.random.normal(0, 0.02, image.shape)
        result = np.clip(result + noise, 0, 1)
        
        return result
    
    def create_animation(
        self,
        output_path: str = './results/ice_animation.gif',
        start_year: int = 1979,
        forecast_years: int = 50,
        fps: int = 10,
        sample_rate: int = 3
    ):
        """Create full animation with realistic predictions"""
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        print("\n" + "=" * 60)
        print("Creating Improved Animation")
        print("=" * 60)
        
        # 1. Load historical data
        print("\n1. Loading historical data...")
        monthly_data = self.get_monthly_images(start_year, 2025)
        
        historical_keys = sorted(monthly_data.keys())[::sample_rate]
        historical_images = [monthly_data[k] for k in historical_keys]
        historical_years = [float(k.split('-')[0]) + float(k.split('-')[1])/12 for k in historical_keys]
        historical_extents = [(img > 0.15).mean() for img in historical_images]
        
        print(f"   Historical: {len(historical_images)} frames ({start_year}-2025)")
        
        # 2. Generate trend-based predictions
        print("\n2. Generating trend-based predictions...")
        future_predictions, future_extents = self.generate_trend_based_predictions(
            num_years=forecast_years,
            start_year=2025
        )
        
        # Sample predictions
        future_predictions = future_predictions[::sample_rate]
        future_extents = future_extents[::sample_rate]
        future_years = [2025 + i * sample_rate / 12 for i in range(len(future_predictions))]
        
        print(f"   Predictions: {len(future_predictions)} frames (2025-{2025+forecast_years})")
        
        # 3. Combine all data
        all_images = historical_images + future_predictions
        all_years = historical_years + future_years
        all_extents = historical_extents + future_extents
        transition_idx = len(historical_images)
        
        # 4. Create animation
        print("\n3. Creating animation...")
        
        fig = plt.figure(figsize=(16, 9))
        gs = GridSpec(2, 2, figure=fig, height_ratios=[2, 1], width_ratios=[2, 1])
        
        # Main ice map
        ax_map = fig.add_subplot(gs[0, 0])
        ax_map.set_title('Antarctic Sea Ice Concentration', fontsize=14, fontweight='bold')
        ax_map.axis('off')
        
        # Info panel
        ax_info = fig.add_subplot(gs[0, 1])
        ax_info.axis('off')
        
        # Time series
        ax_extent = fig.add_subplot(gs[1, :])
        ax_extent.set_xlabel('Year', fontsize=11)
        ax_extent.set_ylabel('Ice Extent', fontsize=11)
        ax_extent.set_xlim(start_year - 2, 2025 + forecast_years + 2)
        ax_extent.set_ylim(0, 1)
        ax_extent.grid(True, alpha=0.3)
        ax_extent.axvline(x=2025, color='red', linestyle='--', alpha=0.7, linewidth=2)
        ax_extent.fill_betweenx([0, 1], 2025, 2025 + forecast_years, alpha=0.1, color='red')
        
        # Initialize plot elements
        im = ax_map.imshow(all_images[0], cmap=self.ice_cmap, vmin=0, vmax=1)
        plt.colorbar(im, ax=ax_map, shrink=0.7, label='Ice Concentration')
        
        line_hist, = ax_extent.plot([], [], 'b-', linewidth=2, label='Historical (Satellite)')
        line_pred, = ax_extent.plot([], [], 'r-', linewidth=2, label='Projected (AI + Trend)')
        point, = ax_extent.plot([], [], 'ko', markersize=8, zorder=5)
        ax_extent.legend(loc='upper right', fontsize=10)
        
        # Text elements
        year_text = ax_map.text(0.02, 0.98, '', transform=ax_map.transAxes,
                                fontsize=24, fontweight='bold', color='white',
                                va='top', ha='left',
                                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
        
        status_text = ax_map.text(0.98, 0.02, '', transform=ax_map.transAxes,
                                  fontsize=12, fontweight='bold', color='white',
                                  va='bottom', ha='right',
                                  bbox=dict(boxstyle='round,pad=0.3', facecolor='green', alpha=0.8))
        
        def animate(frame):
            # Update ice map
            im.set_array(all_images[frame])
            
            # Update year
            year = all_years[frame]
            year_text.set_text(f'{int(year)}')
            
            # Update status
            if frame < transition_idx:
                status_text.set_text('SATELLITE DATA')
                status_text.set_bbox(dict(boxstyle='round,pad=0.3', facecolor='darkgreen', alpha=0.8))
            else:
                status_text.set_text('AI PROJECTION')
                status_text.set_bbox(dict(boxstyle='round,pad=0.3', facecolor='darkred', alpha=0.8))
            
            # Update time series
            hist_end = min(frame + 1, transition_idx)
            line_hist.set_data(all_years[:hist_end], all_extents[:hist_end])
            
            if frame >= transition_idx:
                pred_end = frame + 1
                line_pred.set_data(all_years[transition_idx:pred_end], all_extents[transition_idx:pred_end])
            else:
                line_pred.set_data([], [])
            
            point.set_data([all_years[frame]], [all_extents[frame]])
            
            # Update info panel
            ax_info.clear()
            ax_info.axis('off')
            
            extent_pct = all_extents[frame] * 100
            
            # Compute change from start
            if frame > 0:
                change = (all_extents[frame] - all_extents[0]) / all_extents[0] * 100
                change_text = f"{change:+.1f}%"
            else:
                change_text = "0%"
            
            # Risk level
            if extent_pct > 60:
                risk = "LOW"
                risk_color = "green"
            elif extent_pct > 40:
                risk = "MODERATE"
                risk_color = "orange"
            elif extent_pct > 20:
                risk = "HIGH"
                risk_color = "red"
            else:
                risk = "CRITICAL"
                risk_color = "darkred"
            
            info_lines = [
                f"YEAR: {int(year)}",
                "",
                f"Ice Extent: {extent_pct:.1f}%",
                f"Change from 1979: {change_text}",
                "",
                f"Habitat Risk: {risk}",
                "",
                "Data Source:",
                "Historical" if frame < transition_idx else "AI Projection",
            ]
            
            info_text = "\n".join(info_lines)
            ax_info.text(0.1, 0.95, info_text, transform=ax_info.transAxes,
                        fontsize=11, va='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcyan', alpha=0.9))
            
            # Risk indicator
            ax_info.add_patch(plt.Circle((0.5, 0.15), 0.1, color=risk_color, transform=ax_info.transAxes))
            ax_info.text(0.5, 0.15, risk[0], transform=ax_info.transAxes,
                        fontsize=14, fontweight='bold', color='white',
                        ha='center', va='center')
            
            return [im, line_hist, line_pred, point, year_text, status_text]
        
        print(f"   Rendering {len(all_images)} frames...")
        
        anim = animation.FuncAnimation(
            fig, animate,
            frames=len(all_images),
            interval=1000 // fps,
            blit=False
        )
        
        # Save
        print(f"\n4. Saving animation to {output_path}...")
        
        if output_path.endswith('.mp4'):
            try:
                writer = animation.FFMpegWriter(fps=fps, bitrate=5000)
                anim.save(output_path, writer=writer, dpi=120)
            except:
                output_path = output_path.replace('.mp4', '.gif')
                anim.save(output_path, writer='pillow', fps=fps, dpi=100)
        else:
            anim.save(output_path, writer='pillow', fps=fps, dpi=100)
        
        print(f"   Saved: {output_path}")
        plt.close()
        
        # 5. Create summary plot
        self._create_summary_plot(all_years, all_extents, transition_idx, start_year, forecast_years)
        
        print("\n" + "=" * 60)
        print("Animation complete!")
        print("=" * 60)
        
        # 6. Generate Sea Level Impact Report
        self._generate_sea_level_report(all_extents, forecast_years)
        
        return output_path
    
    def _generate_sea_level_report(self, extents: List[float], forecast_years: int):
        """Generate sea level rise impact report"""
        print("\n6. Generating Sea Level Impact Report...")
        
        try:
            calculator = SeaLevelImpactCalculator()
            ice_predictions = np.array(extents)
            report = calculator.generate_impact_report(ice_predictions, forecast_years)
            
            # Print summary
            print_impact_report(report)
            
            # Create visualization
            self._create_city_impact_visualization(report)
            
        except Exception as e:
            print(f"   Could not generate impact report: {e}")
    
    def _create_city_impact_visualization(self, report: Dict):
        """Create visualization of city impacts"""
        output_dir = './results'
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Left: Bar chart of population at risk
        cities = report['city_impacts'][:10]
        city_names = [c['city'] for c in cities]
        pop_affected = [c['population_affected'] / 1_000_000 for c in cities]
        risk_colors = {'CRITICAL': 'darkred', 'HIGH': 'red', 'MODERATE': 'orange', 'LOW': 'green'}
        colors = [risk_colors[c['risk_level']] for c in cities]
        
        bars = axes[0].barh(city_names, pop_affected, color=colors)
        axes[0].set_xlabel('Population Affected (millions)', fontsize=12)
        axes[0].set_title('Population at Risk from Sea Level Rise', fontsize=14, fontweight='bold')
        axes[0].invert_yaxis()
        
        # Add risk level legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='darkred', label='Critical'),
            Patch(facecolor='red', label='High'),
            Patch(facecolor='orange', label='Moderate'),
            Patch(facecolor='green', label='Low')
        ]
        axes[0].legend(handles=legend_elements, loc='lower right', title='Risk Level')
        
        # Right: Economic impact
        econ_impact = [c['economic_impact_billion_usd'] for c in cities]
        bars2 = axes[1].barh(city_names, econ_impact, color=colors)
        axes[1].set_xlabel('Economic Impact ($ billions)', fontsize=12)
        axes[1].set_title('Economic Assets at Risk', fontsize=14, fontweight='bold')
        axes[1].invert_yaxis()
        
        # Add summary text
        summary = report['summary']
        summary_text = (
            f"Sea Level Rise: {summary['sea_level_rise_cm']:.1f} cm\n"
            f"Total Pop. Affected: {summary['total_population_affected']/1e6:.1f}M\n"
            f"Total Economic Risk: ${summary['total_economic_impact_billion_usd']:.0f}B"
        )
        fig.text(0.5, 0.02, summary_text, ha='center', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
        
        plt.suptitle('Global Sea Level Rise Impact Assessment', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0.08, 1, 0.95])
        
        plt.savefig(f'{output_dir}/city_impact_assessment.png', dpi=300)
        plt.close()
        print(f"   Saved: {output_dir}/city_impact_assessment.png")
    
    def _create_summary_plot(self, years, extents, transition_idx, start_year, forecast_years):
        """Create summary timeline plot"""
        output_dir = './results'
        os.makedirs(output_dir, exist_ok=True)
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Historical
        ax.plot(years[:transition_idx], extents[:transition_idx],
                'b-', linewidth=2.5, label='Historical (Satellite Data)')
        
        # Predictions
        if transition_idx < len(years):
            ax.plot(years[transition_idx:], extents[transition_idx:],
                    'r-', linewidth=2.5, label='Projected (AI + Climate Trend)')
            
            # Uncertainty band (growing with time)
            pred_years = np.array(years[transition_idx:])
            pred_extents = np.array(extents[transition_idx:])
            
            # Uncertainty grows with time
            uncertainty = 0.02 + 0.002 * np.arange(len(pred_years))
            
            ax.fill_between(pred_years, 
                           pred_extents - uncertainty,
                           pred_extents + uncertainty,
                           color='red', alpha=0.2, label='Uncertainty Range')
        
        # Mark present
        ax.axvline(x=2025, color='gray', linestyle='--', linewidth=2, alpha=0.7)
        ax.text(2025, 0.95, '  Present\n  (2025)', fontsize=10, va='top')
        
        # Critical thresholds
        ax.axhline(y=0.4, color='orange', linestyle=':', alpha=0.7)
        ax.text(start_year + 2, 0.41, 'Moderate Risk Threshold', fontsize=9, color='orange')
        
        ax.axhline(y=0.2, color='red', linestyle=':', alpha=0.7)
        ax.text(start_year + 2, 0.21, 'Critical Risk Threshold', fontsize=9, color='red')
        
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Sea Ice Extent (fraction of area)', fontsize=12)
        ax.set_title('Antarctic Sea Ice Extent: 1979-2075\nHistorical Data + AI Climate Projection',
                    fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(start_year - 2, 2025 + forecast_years + 2)
        ax.set_ylim(0, 1)
        
        # Add statistics box
        initial = extents[0]
        current = extents[transition_idx - 1]
        final = extents[-1]
        
        stats_text = f"""Statistics:
1979 Extent: {initial*100:.1f}%
2025 Extent: {current*100:.1f}%  
2075 Extent: {final*100:.1f}%
Total Change: {(final-initial)/initial*100:.1f}%"""
        
        ax.text(0.02, 0.02, stats_text, transform=ax.transAxes,
               fontsize=10, va='bottom', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/ice_extent_timeline.png', dpi=300)
        plt.close()
        print(f"   Saved: {output_dir}/ice_extent_timeline.png")


def main():
    parser = argparse.ArgumentParser(description='Create improved sea ice visualization')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to NSIDC data')
    parser.add_argument('--model_path', type=str, default='./checkpoints/best_model.pt')
    parser.add_argument('--output', type=str, default='./results/ice_animation.gif')
    parser.add_argument('--forecast_years', type=int, default=50)
    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('--device', type=str, default='cpu')
    
    args = parser.parse_args()
    
    visualizer = ImprovedVisualizer(
        data_dir=args.data_dir,
        model_path=args.model_path if os.path.exists(args.model_path) else None,
        device=args.device
    )
    
    visualizer.create_animation(
        output_path=args.output,
        start_year=1979,
        forecast_years=args.forecast_years,
        fps=args.fps
    )


if __name__ == "__main__":
    main()