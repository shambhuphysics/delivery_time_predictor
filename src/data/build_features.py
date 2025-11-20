"""Feature engineering for delivery time prediction."""

import logging
import pandas as pd
import numpy as np
import torch
import yaml
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FeatureEngineering:
    """Build domain-specific features for delivery prediction."""
    
    def __init__(self, config_path: str = "configs/data.yaml"):
        self.config = self._load_config(config_path)
        self.project_root = Path.cwd()
        self.processed_dir = self.project_root / self.config['paths']['processed']
        self.feature_config = self.config['features']
        self.splits = ["train", "val", "test"]
        self.mean = None
        self.std = None
    
    def _load_config(self, config_path: str) -> dict:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def rush_hour_feature(self, hours: np.ndarray, weekends: np.ndarray) -> np.ndarray:
        """Create rush hour binary feature."""
        is_morning = (hours > 8.0) & (hours < 10.0)
        is_evening = (hours >= 16.0) & (hours < 19.0)
        is_weekday = (weekends == 0)
        return (is_weekday & (is_morning | is_evening)).astype(np.float32)
    
    def cyclical_time_features(self, hours: np.ndarray) -> tuple:
        """Create sin/cos encoding of hour."""
        radians = 2 * np.pi * hours / 24.0
        return np.sin(radians).astype(np.float32), np.cos(radians).astype(np.float32)
    
    def build_features(self, df: pd.DataFrame) -> tuple:
        """Build all features from dataframe."""
        # Extract raw features
        distance = df[self.feature_config['distance_col']].values.astype(np.float32)
        hour = df[self.feature_config['hour_col']].values.astype(np.float32)
        weekend = df[self.feature_config['weekend_col']].values.astype(np.float32)
        time_min = df[self.feature_config['target_col']].values.astype(np.float32)
        
        # Engineered features
        features_list = [distance, hour]
        
        if self.feature_config['create_speed']:
            speed = np.divide(distance, time_min / 60, out=np.zeros_like(distance), where=(time_min > 0))
            features_list.append(speed)
        
        features_list.append(weekend)
        
        if self.feature_config['create_rush_hour']:
            rush = self.rush_hour_feature(hour, weekend)
            features_list.append(rush)
        
        if self.feature_config['create_interactions']:
            interaction = rush * weekend if self.feature_config['create_rush_hour'] else weekend * hour
            features_list.append(interaction)
        
        if self.feature_config['create_cyclical_time']:
            cyc_sin, cyc_cos = self.cyclical_time_features(hour)
            features_list.extend([cyc_sin, cyc_cos])
        
        features = np.stack(features_list, axis=1)
        targets = time_min.reshape(-1, 1)
        
        return features, targets
    
    def fit_normalization(self, features: np.ndarray):
        """Fit normalization on training features."""
        norm_idx = self.feature_config['normalize_indices']
        self.mean = features[:, norm_idx].mean(axis=0)
        self.std = features[:, norm_idx].std(axis=0)
        self.std[self.std == 0] = 1.0
        logger.info(f"✓ Fitted normalization on {len(norm_idx)} features")
    
    def apply_normalization(self, features: np.ndarray) -> np.ndarray:
        """Apply fitted normalization."""
        features_norm = features.copy()
        norm_idx = self.feature_config['normalize_indices']
        features_norm[:, norm_idx] = (features[:, norm_idx] - self.mean) / self.std
        return features_norm
    
    def save_normalization_params(self):
        """Save mean and std for inference."""
        np.save(self.processed_dir / "feature_mean.npy", self.mean)
        np.save(self.processed_dir / "feature_std.npy", self.std)
        logger.info("✓ Saved normalization parameters")
    
    def save_tensors(self, features: np.ndarray, targets: np.ndarray, split: str):
        """Save features and targets as PyTorch tensors."""
        feats_tensor = torch.tensor(features, dtype=torch.float32)
        targets_tensor = torch.tensor(targets, dtype=torch.float32)
        
        torch.save(feats_tensor, self.processed_dir / f"features_{split}.pt")
        torch.save(targets_tensor, self.processed_dir / f"targets_{split}.pt")
        
        logger.info(f"✓ Saved {split}: features {feats_tensor.shape}, targets {targets_tensor.shape}")
    
    def run(self, source: str = "delivery"):
        """Run complete feature engineering pipeline."""
        # Process training set first to fit normalization
        source_config = self.config['data_sources'][source]
        train_path = self.processed_dir / source_config['filename'].replace('.csv', '_train.csv')
        
        logger.info(f"Processing training set: {train_path}")
        df_train = pd.read_csv(train_path)
        train_feats, train_targets = self.build_features(df_train)
        
        # Fit normalization
        self.fit_normalization(train_feats)
        self.save_normalization_params()
        
        # Process all splits
        for split in self.splits:
            split_path = self.processed_dir / source_config['filename'].replace('.csv', f'_{split}.csv')
            logger.info(f"Processing {split}: {split_path}")
            
            df = pd.read_csv(split_path)
            features, targets = self.build_features(df)
            features_norm = self.apply_normalization(features)
            self.save_tensors(features_norm, targets, split)
        
        logger.info("="*50)
        logger.info("✓ Feature engineering completed")
        logger.info("="*50)


def main():
    """Run feature engineering pipeline."""
    try:
        engineer = FeatureEngineering()
        engineer.run()
        return 0
    except Exception as e:
        logger.error(f"✗ Feature engineering failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
