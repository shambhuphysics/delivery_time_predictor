import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, mock_open
from src.data.build_features import FeatureEngineering


@pytest.fixture
def mock_config():
    return """
data_sources:
  delivery:
    filename: "test.csv"
paths:
  processed: "data/processed"
features:
  distance_col: "distance_miles"
  hour_col: "time_of_day_hours"
  weekend_col: "is_weekend"
  target_col: "delivery_time_minutes"
  create_rush_hour: true
  create_speed: true
  create_cyclical_time: true
  create_interactions: true
  normalize_indices: [0, 1, 2, 6, 7]
"""


@pytest.fixture
def sample_df():
    """Sample delivery dataframe."""
    return pd.DataFrame({
        'distance_miles': [5.0, 10.0, 15.0],
        'time_of_day_hours': [9.0, 17.0, 12.0],
        'is_weekend': [0.0, 0.0, 1.0],
        'delivery_time_minutes': [30.0, 45.0, 50.0]
    })


def test_rush_hour_feature(mock_config, sample_df):
    """Test rush hour feature creation."""
    with patch('builtins.open', mock_open(read_data=mock_config)):
        engineer = FeatureEngineering()
        hours = sample_df['time_of_day_hours'].values
        weekends = sample_df['is_weekend'].values
        rush = engineer.rush_hour_feature(hours, weekends)
        
        assert rush[0] == 1.0  # 9am weekday
        assert rush[1] == 1.0  # 5pm weekday
        assert rush[2] == 0.0  # weekend


def test_cyclical_features(mock_config, sample_df):
    """Test cyclical time encoding."""
    with patch('builtins.open', mock_open(read_data=mock_config)):
        engineer = FeatureEngineering()
        hours = sample_df['time_of_day_hours'].values
        sin_vals, cos_vals = engineer.cyclical_time_features(hours)
        
        assert sin_vals.shape == (3,)
        assert cos_vals.shape == (3,)
        assert -1 <= sin_vals.min() <= 1
        assert -1 <= cos_vals.min() <= 1


def test_build_features(mock_config, sample_df):
    """Test complete feature building."""
    with patch('builtins.open', mock_open(read_data=mock_config)):
        engineer = FeatureEngineering()
        features, targets = engineer.build_features(sample_df)
        
        assert features.shape[0] == 3
        assert features.shape[1] == 8
        assert targets.shape == (3, 1)
        assert not np.isnan(features).any()
        assert not np.isinf(features).any()


def test_normalization(mock_config, sample_df):
    """Test normalization fitting and applying."""
    with patch('builtins.open', mock_open(read_data=mock_config)):
        engineer = FeatureEngineering()
        features, _ = engineer.build_features(sample_df)
        
        engineer.fit_normalization(features)
        assert engineer.mean is not None
        assert engineer.std is not None
        
        features_norm = engineer.apply_normalization(features)
        norm_idx = [0, 1, 2, 6, 7]
        for idx in norm_idx:
            assert np.abs(features_norm[:, idx].mean()) < 1.0
