import pytest
import pandas as pd
from unittest.mock import patch, mock_open
from src.data.validation import DataValidation


@pytest.fixture
def mock_config():
    """Mock config."""
    return """
data_sources:
  phonopy:
    filename: "test.csv"
paths:
  raw: "data/raw"
validation:
  missing_value_threshold: 0.5
  allow_duplicates: false
  allow_negative_values: true
  allow_infinite_values: false
  min_rows: 1
  min_columns: 1
"""


@pytest.fixture
def sample_df():
    """Sample valid DataFrame."""
    return pd.DataFrame({
        'col1': [1, 2, 3, 4],
        'col2': [5, 6, 7, 8]
    })


def test_validate_shape(mock_config, sample_df):
    """Test shape validation."""
    with patch('builtins.open', mock_open(read_data=mock_config)):
        validator = DataValidation()
        assert validator.validate_shape(sample_df) == True


def test_validate_missing_values(mock_config, sample_df):
    """Test missing value validation."""
    with patch('builtins.open', mock_open(read_data=mock_config)):
        validator = DataValidation()
        assert validator.validate_missing_values(sample_df) == True


def test_validate_duplicates_fails(mock_config):
    """Test duplicate detection."""
    df = pd.DataFrame({'col1': [1, 1, 2], 'col2': [3, 3, 4]})
    with patch('builtins.open', mock_open(read_data=mock_config)):
        validator = DataValidation()
        with pytest.raises(ValueError, match="duplicate"):
            validator.validate_duplicates(df)


def test_validate_infinite_values_fails(mock_config):
    """Test infinite value detection."""
    df = pd.DataFrame({'col1': [1, float('inf'), 3]})
    with patch('builtins.open', mock_open(read_data=mock_config)):
        validator = DataValidation()
        with pytest.raises(ValueError, match="infinite"):
            validator.validate_value_ranges(df)
