import pytest
import torch
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock
from src.models.dataloader import DataLoaderFactory, get_loaders


@pytest.fixture
def mock_config():
    return """
model:
  input_dim: 8
training:
  batch_size: 32
data:
  features_dir: "data/processed"
  shuffle_train: true
  num_workers: 0
"""


@pytest.fixture
def mock_tensors(tmp_path):
    """Create mock tensor files."""
    data_dir = tmp_path / "data" / "processed"
    data_dir.mkdir(parents=True)
    
    # Create sample tensors
    X = torch.randn(100, 8)
    y = torch.randn(100, 1)
    
    torch.save(X, data_dir / "features_train.pt")
    torch.save(y, data_dir / "targets_train.pt")
    torch.save(X[:20], data_dir / "features_val.pt")
    torch.save(y[:20], data_dir / "targets_val.pt")
    torch.save(X[:20], data_dir / "features_test.pt")
    torch.save(y[:20], data_dir / "targets_test.pt")
    
    return data_dir


def test_dataloader_creation(mock_config, mock_tensors, tmp_path):
    """Test DataLoader creation."""
    with patch('builtins.open', mock_open(read_data=mock_config)):
        with patch('pathlib.Path.cwd', return_value=tmp_path):
            factory = DataLoaderFactory()
            train_loader = factory.load_split('train')
            
            assert len(train_loader.dataset) == 100
            assert train_loader.batch_size == 32


def test_batch_shape(mock_config, mock_tensors, tmp_path):
    """Test batch shapes are correct."""
    with patch('builtins.open', mock_open(read_data=mock_config)):
        with patch('pathlib.Path.cwd', return_value=tmp_path):
            factory = DataLoaderFactory()
            train_loader = factory.load_split('train')
            
            X_batch, y_batch = next(iter(train_loader))
            assert X_batch.shape[1] == 8  # Feature dimension
            assert y_batch.shape[1] == 1  # Target dimension


def test_get_loaders_function(mock_config, mock_tensors, tmp_path):
    """Test convenience function."""
    with patch('builtins.open', mock_open(read_data=mock_config)):
        with patch('pathlib.Path.cwd', return_value=tmp_path):
            train, val, test = get_loaders()
            
            assert len(train.dataset) == 100
            assert len(val.dataset) == 20
            assert len(test.dataset) == 20
