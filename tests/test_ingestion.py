# tests/test_ingestion.py

import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch
from src.data.ingestion import DataIngestion, main


@pytest.fixture
def ingestion():
    """Create DataIngestion instance."""
    return DataIngestion()


@pytest.fixture
def mock_csv_response():
    """Mock HTTP response with CSV data."""
    mock = Mock()
    mock.content = b"col1,col2\n1,2\n3,4\n"
    mock.status_code = 200
    return mock


# ============================================================================
# TESTS
# ============================================================================

def test_initialization(ingestion):
    """Verify raw data directory is created."""
    assert ingestion.raw_data_dir.exists()
    assert ingestion.raw_data_dir.name == "raw"


def test_github_url_conversion(ingestion, mock_csv_response):
    """Verify GitHub URL converts to raw URL."""
    with patch('requests.get', return_value=mock_csv_response) as mock_get:
        ingestion.download_from_github(
            "https://github.com/user/repo/blob/main/file.csv",
            "test.csv"
        )
        called_url = mock_get.call_args[0][0]
        assert "raw.githubusercontent.com" in called_url
        assert "/blob/" not in called_url


def test_download_creates_file(ingestion, mock_csv_response):
    """Verify file is downloaded and saved."""
    with patch('requests.get', return_value=mock_csv_response):
        result = ingestion.download_from_github(
            "https://github.com/user/repo/blob/main/file.csv",
            "test.csv"
        )
        assert result.exists()
        df = pd.read_csv(result)
        assert df.shape == (2, 2)


def test_download_handles_errors(ingestion):
    """Verify HTTP errors are raised."""
    with patch('requests.get', side_effect=Exception("Network error")):
        with pytest.raises(Exception):
            ingestion.download_from_github(
                "https://github.com/user/repo/blob/main/file.csv",
                "test.csv"
            )


def test_main_success(mock_csv_response):
    """Verify main returns 0 on success."""
    with patch('requests.get', return_value=mock_csv_response):
        assert main() == 0


def test_main_failure():
    """Verify main returns 1 on failure."""
    with patch('requests.get', side_effect=Exception("Error")):
        assert main() == 1
