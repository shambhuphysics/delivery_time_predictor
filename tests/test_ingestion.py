import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
from src.data.ingestion import DataIngestion


@pytest.fixture
def mock_config():
    """Mock config data."""
    return """
data_sources:
  phonopy:
    url: "https://github.com/user/repo/blob/main/test.csv"
    filename: "test.csv"
paths:
  raw: "data/raw"
download:
  timeout_connect: 5
  timeout_read: 30
  enable_versioning: true
"""


@pytest.fixture
def mock_response():
    """Mock HTTP response."""
    mock = Mock()
    mock.content = b"col1,col2\n1,2\n3,4\n"
    mock.status_code = 200
    return mocks


def test_config_loading(mock_config):
    """Test configuration loads correctly."""
    with patch('builtins.open', mock_open(read_data=mock_config)):
        ingestion = DataIngestion()
        assert 'data_sources' in ingestion.config
        assert 'phonopy' in ingestion.config['data_sources']


def test_download_from_github(mock_config, mock_response):
    """Test file download works."""
    with patch('builtins.open', mock_open(read_data=mock_config)):
        with patch('requests.get', return_value=mock_response):
            ingestion = DataIngestion()
            result = ingestion.download_from_github(
                "https://github.com/user/repo/blob/main/test.csv",
                "test.csv"
            )
            assert result.name == "test.csv"


def test_ingest_data(mock_config, mock_response):
    """Test data ingestion works."""
    with patch('builtins.open', mock_open(read_data=mock_config)):
        with patch('requests.get', return_value=mock_response):
            ingestion = DataIngestion()
            result = ingestion.ingest_data("phonopy")
            assert result.exists()
