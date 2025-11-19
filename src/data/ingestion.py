import logging
import requests
import pandas as pd
import yaml
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataIngestion:
    """Handles data ingestion from external sources."""
    
    def __init__(self, config_path: str = "configs/data.yaml"):
        self.config = self._load_config(config_path)
        self.project_root = Path.cwd()
        self.raw_data_dir = self.project_root / self.config['paths']['raw']
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Raw data directory: {self.raw_data_dir}")
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def download_from_github(self, url: str, filename: str) -> Path:
        """Download and save CSV data from GitHub."""
        raw_url = url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
        logger.info(f"Downloading from: {raw_url}")
        
        timeout = (
            self.config['download']['timeout_connect'],
            self.config['download']['timeout_read']
        )
        response = requests.get(raw_url, timeout=timeout)
        response.raise_for_status()
        
        # Save files
        if self.config['download']['enable_versioning']:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = filename.rsplit('.', 1)[0]
            versioned_path = self.raw_data_dir / f"{base_name}_{timestamp}.csv"
            versioned_path.write_bytes(response.content)
        
        latest_path = self.raw_data_dir / filename
        latest_path.write_bytes(response.content)
        
        # Validate
        df = pd.read_csv(latest_path)
        logger.info(f"Downloaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        return latest_path
    
    def ingest_data(self, source: str = "phonopy") -> Path:
        """Ingest data from configured source."""
        source_config = self.config['data_sources'][source]
        return self.download_from_github(
            source_config['url'],
            source_config['filename']
        )


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
