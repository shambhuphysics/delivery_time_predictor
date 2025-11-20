"""DataLoader utilities for model training."""

import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path


class DataLoaderFactory:
    """Creates PyTorch DataLoaders from saved tensors."""
    
    def __init__(self, config_path: str = "configs/model.yaml"):
        self.config = self._load_config(config_path)
        self.project_root = Path.cwd()
        self.data_dir = self.project_root / self.config['data']['features_dir']
        self.batch_size = self.config['training']['batch_size']
        self.num_workers = self.config['data']['num_workers']
    
    def _load_config(self, config_path: str) -> dict:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_split(self, split: str) -> DataLoader:
        """Load a single data split as DataLoader."""
        features_path = self.data_dir / f"features_{split}.pt"
        targets_path = self.data_dir / f"targets_{split}.pt"
        
        X = torch.load(features_path)
        y = torch.load(targets_path)
        
        shuffle = (split == 'train') and self.config['data']['shuffle_train']
        
        dataset = TensorDataset(X, y)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers
        )
        
        return loader
    
    def get_loaders(self) -> tuple:
        """Get train, val, and test DataLoaders."""
        train_loader = self.load_split('train')
        val_loader = self.load_split('val')
        test_loader = self.load_split('test')
        
        return train_loader, val_loader, test_loader


def get_loaders(batch_size: int = None, config_path: str = "configs/model.yaml") -> tuple:
    """Convenience function to get DataLoaders."""
    factory = DataLoaderFactory(config_path)
    
    if batch_size is not None:
        factory.batch_size = batch_size
    
    return factory.get_loaders()
