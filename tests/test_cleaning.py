import pytest
import pandas as pd
from unittest.mock import patch, mock_open
from src.data.cleaning import DataCleaning


@pytest.fixture
def mock_config():
    return """
data_sources:
  phonopy:
    filename: "test.csv"
paths:
  raw: "data/raw"
  processed: "data/processed"
cleaning:
  missing_row_threshold: 0.5
  cap_outliers: true
  iqr_multiplier: 1.5
"""


@pytest.fixture
def sample_df_with_issues():
    return pd.DataFrame({
        'col1': [1, 2, 2, 100, 5],  # Has duplicate and outlier
        'col2': [1, 2, 2, 3, 4]
    })


def test_remove_duplicates(mock_config, sample_df_with_issues):
    """Test duplicate removal."""
    with patch('builtins.open', mock_open(read_data=mock_config)):
        cleaner = DataCleaning()
        result = cleaner.remove_duplicates(sample_df_with_issues)
        assert len(result) == 4  # One duplicate removed


def test_cap_outliers(mock_config):
    """Test outlier capping."""
    df = pd.DataFrame({'col1': [1, 2, 3, 4, 100]})
    with patch('builtins.open', mock_open(read_data=mock_config)):
        cleaner = DataCleaning()
        result = cleaner.cap_outliers(df)
        assert result['col1'].max() < 100  # Outlier capped
