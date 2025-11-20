import pytest
import torch
from unittest.mock import patch, mock_open
from src.models.model1.model import DeliveryTimeNN


@pytest.fixture
def mock_config():
    return """
model:
  input_dim: 8
  hidden_dims: [64, 32]
  dropout: 0.2
training:
  batch_size: 32
data:
  features_dir: "data/processed"
  num_workers: 0
logging:
  save_dir: "models"
"""


def test_model_forward(mock_config):
    """Test model forward pass."""
    with patch('builtins.open', mock_open(read_data=mock_config)):
        model = DeliveryTimeNN()
        x = torch.randn(10, 8)
        output = model(x)
        
        assert output.shape == (10, 1)
        assert not torch.isnan(output).any()


def test_model_trainable(mock_config):
    """Test model parameters are trainable."""
    with patch('builtins.open', mock_open(read_data=mock_config)):
        model = DeliveryTimeNN()
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 0
        assert all(p.requires_grad for p in model.parameters())
