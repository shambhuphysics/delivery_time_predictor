"""PyTorch model for delivery time prediction."""

import torch
import torch.nn as nn
import yaml
from pathlib import Path


class DeliveryTimeNN(nn.Module):
    """MLP for delivery time regression."""
    
    def __init__(self, config_path: str = "configs/model.yaml"):
        super().__init__()
        
        # Load config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        model_cfg = config['model']
        input_dim = model_cfg['input_dim']
        hidden_dims = model_cfg['hidden_dims']
        dropout = model_cfg.get('dropout', 0.2)
        
        # Build network
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dims[1], 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Better weight initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.model(x)
