"""
Data loading and preprocessing for NSIDC Antarctic Sea-Ice Images
Handles multiple image formats and creates temporal sequences for training
"""
import os
import re
import glob
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Dict
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import warnings

class NSIDCDataLoader:
    """
    Loader for NSIDC Sea Ice Index daily images
    Expected directory structure:
    data_dir/
        YYYY/
            MM_Mon/
                S_YYYYMMDD_concentration_v3.0.png
        OR flat structure with dated filenames
    """
    
    def __init__(self, data_dir: str, image_size: Tuple[int, int] = (256, 256)):
        self.data_dir = data_dir
        self.image_size = image_size
        self.image_cache: Dict[str, np.ndarray] = {}
        
        # Find all images and create date index
        self.image_index = self._build_image_index()
        print(f"Found {len(self.image_index)} dated images")
        
    def _build_image_index(self) -> Dict[str, str]:
        """Build a dictionary mapping dates to image paths"""
        index = {}
        
        # Common NSIDC filename patterns
        patterns = [
            r'S_(\d{8}).*\.png',  # S_YYYYMMDD_*.png
            r'S_(\d{8}).*\.jpg',
            r'(\d{8}).*\.png',
            r'(\d{4})(\d{2})(\d{2}).*\.(png|jpg|tif)',
        ]
        
        # Search recursively for all image files
        for ext in ['png', 'jpg', 'jpeg', 'tif', 'tiff']:
            for filepath in glob.glob(os.path.join(self.data_dir, '**', f'*.{ext}'), recursive=True):
                filename = os.path.basename(filepath)
                
                # Try to extract date from filename
                for pattern in patterns:
                    match = re.search(pattern, filename, re.IGNORECASE)
                    if match:
                        if len(match.groups()) >= 3:
                            # YYYY, MM, DD separate groups
                            date_str = f"{match.group(1)}{match.group(2)}{match.group(3)}"
                        else:
                            date_str = match.group(1)
                        
                        try:
                            # Validate date
                            datetime.strptime(date_str, '%Y%m%d')
                            index[date_str] = filepath
                            break
                        except ValueError:
                            continue
        
        return dict(sorted(index.items()))
    
    def load_image(self, date_str: str) -> Optional[np.ndarray]:
        """Load and preprocess a single image"""
        if date_str not in self.image_index:
            return None
            
        if date_str in self.image_cache:
            return self.image_cache[date_str]
        
        filepath = self.image_index[date_str]
        
        try:
            img = Image.open(filepath).convert('L')  # Convert to grayscale
            img = img.resize(self.image_size, Image.BILINEAR)
            arr = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0, 1]
            
            # Cache if memory allows
            if len(self.image_cache) < 1000:
                self.image_cache[date_str] = arr
                
            return arr
        except Exception as e:
            warnings.warn(f"Error loading {filepath}: {e}")
            return None
    
    def get_date_range(self) -> Tuple[str, str]:
        """Return the date range of available data"""
        dates = list(self.image_index.keys())
        return dates[0], dates[-1]
    
    def get_available_dates(self) -> List[str]:
        """Return list of all available dates"""
        return list(self.image_index.keys())


class SeaIceSequenceDataset(Dataset):
    """
    Dataset that creates temporal sequences for training
    Returns: (input_sequence, target_sequence, dates)
    """
    
    def __init__(
        self,
        data_loader: NSIDCDataLoader,
        sequence_length: int = 365,
        prediction_horizon: int = 30,
        start_year: int = 1979,
        end_year: int = 2020,
        stride: int = 7,  # Sample every N days for efficiency
        augment: bool = False
    ):
        self.data_loader = data_loader
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.stride = stride
        self.augment = augment
        
        # Filter dates within year range
        self.available_dates = [
            d for d in data_loader.get_available_dates()
            if start_year <= int(d[:4]) <= end_year
        ]
        
        # Create valid sequence start indices
        self.valid_starts = self._find_valid_sequences()
        print(f"Created {len(self.valid_starts)} valid sequences for years {start_year}-{end_year}")
        
        # Transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
    def _find_valid_sequences(self) -> List[int]:
        """Find indices where we have enough consecutive data"""
        valid = []
        total_needed = self.sequence_length + self.prediction_horizon
        
        for i in range(0, len(self.available_dates) - total_needed, self.stride):
            # Check if we have enough data points
            start_date = datetime.strptime(self.available_dates[i], '%Y%m%d')
            end_date = datetime.strptime(self.available_dates[min(i + total_needed, len(self.available_dates) - 1)], '%Y%m%d')
            
            # Allow some gaps (use interpolation later)
            expected_days = (end_date - start_date).days
            if expected_days <= total_needed * 1.5:  # Allow 50% gaps
                valid.append(i)
                
        return valid
    
    def _get_sequence(self, start_idx: int, length: int) -> Tuple[torch.Tensor, List[str]]:
        """Get a sequence of images, handling missing dates with interpolation"""
        dates = self.available_dates[start_idx:start_idx + length * 2][:length]
        images = []
        valid_dates = []
        
        for date in dates:
            img = self.data_loader.load_image(date)
            if img is not None:
                images.append(img)
                valid_dates.append(date)
        
        if len(images) < length // 2:
            # Not enough data, return zeros
            h, w = self.data_loader.image_size
            return torch.zeros(length, 1, h, w), dates
        
        # Stack and interpolate if needed
        images = np.stack(images, axis=0)
        
        # If we don't have enough images, interpolate
        if len(images) < length:
            from scipy.ndimage import zoom
            factor = length / len(images)
            images = zoom(images, (factor, 1, 1), order=1)[:length]
        
        # Convert to tensor: (T, C, H, W)
        tensor = torch.from_numpy(images).unsqueeze(1).float()
        
        return tensor, valid_dates
    
    def __len__(self) -> int:
        return len(self.valid_starts)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        start_idx = self.valid_starts[idx]
        
        # Get input sequence
        input_seq, input_dates = self._get_sequence(start_idx, self.sequence_length)
        
        # Get target sequence
        target_start = start_idx + self.sequence_length
        target_seq, target_dates = self._get_sequence(target_start, self.prediction_horizon)
        
        # Data augmentation
        if self.augment:
            if np.random.rand() > 0.5:
                input_seq = torch.flip(input_seq, dims=[3])  # Horizontal flip
                target_seq = torch.flip(target_seq, dims=[3])
        
        metadata = {
            'input_dates': input_dates,
            'target_dates': target_dates,
        }
        
        return input_seq, target_seq, metadata


class YearlyAggregatedDataset(Dataset):
    """
    Alternative dataset that aggregates data by year
    Better for long-term (50-year) forecasting
    Returns yearly sea-ice statistics
    """
    
    def __init__(
        self,
        data_loader: NSIDCDataLoader,
        start_year: int = 1979,
        end_year: int = 2020,
        samples_per_year: int = 12  # Monthly samples
    ):
        self.data_loader = data_loader
        self.samples_per_year = samples_per_year
        
        # Aggregate data by year
        self.yearly_data = self._aggregate_by_year(start_year, end_year)
        self.years = list(self.yearly_data.keys())
        print(f"Aggregated {len(self.years)} years of data")
        
    def _aggregate_by_year(self, start_year: int, end_year: int) -> Dict[int, np.ndarray]:
        """Aggregate images by year, creating monthly composites"""
        yearly = {}
        
        for year in range(start_year, end_year + 1):
            year_images = []
            
            for month in range(1, 13):
                # Get mid-month image
                day = 15
                date_str = f"{year}{month:02d}{day:02d}"
                
                # Try nearby dates if exact date not available
                img = None
                for offset in range(0, 15):
                    for d in [day + offset, day - offset]:
                        if 1 <= d <= 28:
                            try_date = f"{year}{month:02d}{d:02d}"
                            img = self.data_loader.load_image(try_date)
                            if img is not None:
                                break
                    if img is not None:
                        break
                
                if img is not None:
                    year_images.append(img)
                else:
                    # Use placeholder
                    h, w = self.data_loader.image_size
                    year_images.append(np.zeros((h, w), dtype=np.float32))
            
            if len(year_images) == 12:
                yearly[year] = np.stack(year_images, axis=0)
                
        return yearly
    
    def __len__(self) -> int:
        return max(0, len(self.years) - 5)  # Use 5 years to predict 1
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        # Use 5 years of history to predict next year
        input_years = self.years[idx:idx + 5]
        target_year = self.years[idx + 5]
        
        input_data = np.stack([self.yearly_data[y] for y in input_years], axis=0)
        target_data = self.yearly_data[target_year]
        
        # Shape: (5, 12, H, W) -> (5*12, 1, H, W)
        input_tensor = torch.from_numpy(input_data).reshape(-1, 1, *input_data.shape[-2:]).float()
        target_tensor = torch.from_numpy(target_data).unsqueeze(1).float()
        
        return input_tensor, target_tensor, target_year


def create_dataloaders(
    data_dir: str,
    image_size: Tuple[int, int] = (256, 256),
    sequence_length: int = 365,
    prediction_horizon: int = 30,
    batch_size: int = 4,
    num_workers: int = 4,
    train_years: Tuple[int, int] = (1979, 2018),
    val_years: Tuple[int, int] = (2019, 2021),
    test_years: Tuple[int, int] = (2022, 2024),
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test dataloaders"""
    
    # Initialize data loader
    nsidc_loader = NSIDCDataLoader(data_dir, image_size)
    
    # Create datasets
    train_dataset = SeaIceSequenceDataset(
        nsidc_loader,
        sequence_length=sequence_length,
        prediction_horizon=prediction_horizon,
        start_year=train_years[0],
        end_year=train_years[1],
        stride=7,
        augment=True
    )
    
    val_dataset = SeaIceSequenceDataset(
        nsidc_loader,
        sequence_length=sequence_length,
        prediction_horizon=prediction_horizon,
        start_year=val_years[0],
        end_year=val_years[1],
        stride=30,
        augment=False
    )
    
    test_dataset = SeaIceSequenceDataset(
        nsidc_loader,
        sequence_length=sequence_length,
        prediction_horizon=prediction_horizon,
        start_year=test_years[0],
        end_year=test_years[1],
        stride=30,
        augment=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,  # Disabled for Mac/CPU compatibility
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test data loading
    import sys
    
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "./data/nsidc_images"
    
    print("Testing NSIDC Data Loader...")
    loader = NSIDCDataLoader(data_dir)
    
    dates = loader.get_available_dates()
    if dates:
        print(f"Date range: {loader.get_date_range()}")
        print(f"Sample dates: {dates[:5]}...")
        
        # Test loading
        img = loader.load_image(dates[0])
        if img is not None:
            print(f"Image shape: {img.shape}")
            print(f"Value range: [{img.min():.3f}, {img.max():.3f}]")
    else:
        print("No images found. Check your data directory structure.")