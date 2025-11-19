
import logging
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataIngestion:
    """Handles data ingestion from external sources."""
    
    def __init__(self):
        self.project_root = Path(__file__).resolve().parents[2]
        self.raw_data_dir = self.project_root / "data" / "raw"
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Raw data directory: {self.raw_data_dir}")
    
    def download_from_github(self, url: str, filename: str) -> Path:

        # Convert to raw URL
        raw_url = url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
        logger.info(f"Downloading from: {raw_url}")
        
        # Download
        response = requests.get(raw_url, timeout=30)
        response.raise_for_status()
        
        # Save with timestamp for versioning
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        versioned_path = self.raw_data_dir / f"{filename.split('.')[0]}_{timestamp}.csv"
        latest_path = self.raw_data_dir / filename
        
        versioned_path.write_bytes(response.content)
        latest_path.write_bytes(response.content)
        
        # Validate
        df = pd.read_csv(latest_path)
        logger.info(f"Downloaded: {df.shape[0]} rows, {df.shape[1]} columns")
        logger.info(f"Columns: {list(df.columns)}")
        
        return latest_path
    
    def ingest_phonopy_data(self) -> Path:
        """Ingest Phonopy features data."""
        url = "https://github.com/shambhuphysics/Phonopy-scripts/blob/main/data_with_features.csv"
        return self.download_from_github(url, "data_with_features.csv")


def main():
    """Run data ingestion pipeline."""
    try:
        ingestion = DataIngestion()
        data_path = ingestion.ingest_phonopy_data()
        logger.info(f"✓ Data ingestion completed: {data_path}")
        return 0
    except Exception as e:
        logger.error(f"✗ Data ingestion failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
