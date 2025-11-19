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
    
    def __init__(self, project_root=None):
        if project_root is None:
            project_root = Path.cwd() if '__file__' not in globals() else Path(__file__).resolve().parents[2]
        
        self.raw_data_dir = project_root / "data" / "raw"
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Raw data directory: {self.raw_data_dir}")
    
    def download_from_github(self, url: str, filename: str) -> Path:
        """Download and save CSV data from GitHub with versioning."""
        raw_url = url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
        logger.info(f"Downloading from: {raw_url}")
        
        start_time = datetime.now()
        
        # Use separate connect and read timeouts
        response = requests.get(raw_url, timeout=(5, 30))
        response.raise_for_status()
        
        download_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Download completed in {download_time:.2f}s")
        
        # Save with timestamp for versioning
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = filename.rsplit('.', 1)[0]
        versioned_path = self.raw_data_dir / f"{base_name}_{timestamp}.csv"
        latest_path = self.raw_data_dir / filename
        
        content = response.content
        versioned_path.write_bytes(content)
        latest_path.write_bytes(content)
        
        # Validate
        df = pd.read_csv(latest_path)
        logger.info(f"Downloaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        return latest_path
    
    def ingest_data(self) -> Path:
        """Ingest Phonopy features data."""
        url = "https://github.com/shambhuphysics/Phonopy-scripts/blob/main/data_with_features.csv"
        return self.download_from_github(url, "data_with_features.csv")


def main():
    """Run data ingestion pipeline."""
    try:
        ingestion = DataIngestion()
        data_path = ingestion.ingest_data()
        logger.info(f"✓ Data ingestion completed: {data_path}")
        return 0
    except Exception as e:
        logger.error(f"✗ Data ingestion failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
