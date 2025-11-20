import pytest
import torch
from pathlib import Path
from unittest.mock import patch
from src.models.model1.dataloader import DataLoaderFactory, get_loaders


@pytest.fixture
def setup_test_data(tmp_path):
    """Create real test tensor files and config."""
    data_dir = tmp_path / "data" / "processed"
    data_dir.mkdir(parents=True)
    
    X_train = torch.randn(100, 8)
    y_train = torch.randn(100, 1)
    X_val = torch.randn(20, 8)
    y_val = torch.randn(20, 1)
    X_test = torch.randn(20, 8)
    y_test = torch.randn(20, 1)
    
    torch.save(X_train, data_dir / "features_train.pt")
    torch.save(y_train, data_dir / "targets_train.pt")
    torch.save(X_val, data_dir / "features_val.pt")
    torch.save(y_val, data_dir / "targets_val.pt")
    torch.save(X_test, data_dir / "features_test.pt")
    torch.save(y_test, data_dir / "targets_test.pt")
    
    config_path = tmp_path / "configs"
    config_path.mkdir(parents=True)
    config_file = config_path / "model.yaml"
    
    config_content = """
training:
  batch_size: 32
data:
  features_dir: "data/processed"
  shuffle_train: true
  num_workers: 0
"""
    config_file.write_text(config_content)
    
    return tmp_path


def test_dataloader_creation(setup_test_data):
    """Test DataLoader creation."""
    config_path = setup_test_data / "configs" / "model.yaml"
    
    with patch('pathlib.Path.cwd', return_value=setup_test_data):
        factory = DataLoaderFactory(config_path=str(config_path))
        train_loader = factory.load_split('train')
        
        assert len(train_loader.dataset) == 100
        assert train_loader.batch_size == 32


def test_batch_shape(setup_test_data):
    """Test batch shapes are correct."""
    config_path = setup_test_data / "configs" / "model.yaml"
    
    with patch('pathlib.Path.cwd', return_value=setup_test_data):
        factory = DataLoaderFactory(config_path=str(config_path))
        train_loader = factory.load_split('train')
        
        X_batch, y_batch = next(iter(train_loader))
        assert X_batch.shape[1] == 8
        assert y_batch.shape[1] == 1


def test_all_loaders(setup_test_data):
    """Test all three loaders can be created."""
    config_path = setup_test_data / "configs" / "model.yaml"
    
    with patch('pathlib.Path.cwd', return_value=setup_test_data):
        train, val, test = get_loaders(config_path=str(config_path))
        
        assert len(train.dataset) == 100
        assert len(val.dataset) == 20
        assert len(test.dataset) == 20
