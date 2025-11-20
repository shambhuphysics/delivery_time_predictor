import pytest
import pandas as pd
from unittest.mock import patch, mock_open
from src.data.splitting import DataSplit


@pytest.fixture
def mock_config():
    return """
data_sources:
  phonopy:
    filename: "test.csv"
paths:
  processed: "data/processed"
splitting:
  test_size: 0.2
  val_size: 0.2
  random_seed: 42
  shuffle: true
  stratify: false
"""


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'col1': range(100),
        'col2': range(100, 200)
    })


def test_split_ratios(mock_config, sample_df):
    """Test split produces correct ratios."""
    with patch('builtins.open', mock_open(read_data=mock_config)):
        splitter = DataSplit()
        train, val, test = splitter.split_data(sample_df)
        
        total = len(sample_df)
        assert len(test) == pytest.approx(total * 0.2, abs=2)
        assert len(val) == pytest.approx(total * 0.2, abs=2)
        assert len(train) == pytest.approx(total * 0.6, abs=2)


def test_no_data_leakage(mock_config, sample_df):
    """Test no overlap between splits."""
    with patch('builtins.open', mock_open(read_data=mock_config)):
        splitter = DataSplit()
        train, val, test = splitter.split_data(sample_df)
        
        # Check no common indices
        train_idx = set(train.index)
        val_idx = set(val.index)
        test_idx = set(test.index)
        
        assert len(train_idx & val_idx) == 0
        assert len(train_idx & test_idx) == 0
        assert len(val_idx & test_idx) == 0


def test_reproducibility(mock_config, sample_df):
    """Test splits are reproducible with same seed."""
    with patch('builtins.open', mock_open(read_data=mock_config)):
        splitter1 = DataSplit()
        train1, val1, test1 = splitter1.split_data(sample_df.copy())
        
        splitter2 = DataSplit()
        train2, val2, test2 = splitter2.split_data(sample_df.copy())
        
        pd.testing.assert_frame_equal(train1, train2)
        pd.testing.assert_frame_equal(val1, val2)
        pd.testing.assert_frame_equal(test1, test2)
