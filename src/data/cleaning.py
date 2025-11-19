"""
Data Cleaning Module
Follows MLOps principles: config-driven, reusable, reproducible.
"""
import logging
import pandas as pd
import yaml
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataCleaning:
    """Handles data cleaning operations."""
    
    def __init__(self, config_path: str = "configs/data.yaml"):
        self.config = self._load_config(config_path)
        self.project_root = Path.cwd()
        self.raw_data_dir = self.project_root / self.config['paths']['raw']
        self.processed_data_dir = self.project_root / self.config['paths']['processed']
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        self.cleaning_rules = self.config.get('cleaning', {})
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_data(self, source: str = "phonopy") -> pd.DataFrame:
        """Load raw data."""
        source_config = self.config['data_sources'][source]
        data_path = self.raw_data_dir / source_config['filename']
        logger.info(f"Loading: {data_path}")
        return pd.read_csv(data_path)
    
    def handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values based on config."""
        row_threshold = self.cleaning_rules.get('missing_row_threshold', 0.5)
        thresh = int(df.shape[1] * row_threshold)
        
        before_rows = len(df)
        df = df.dropna(thresh=thresh)
        removed_rows = before_rows - len(df)
        
        if removed_rows > 0:
            logger.info(f"✓ Removed {removed_rows} rows with >{row_threshold*100}% missing")
        
        # Fill numeric columns with median
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            logger.info(f"✓ Filled missing values in {len(numeric_cols)} numeric columns")
        
        return df
    
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows."""
        before = len(df)
        df = df.drop_duplicates()
        removed = before - len(df)
        
        if removed > 0:
            logger.info(f"✓ Removed {removed} duplicate rows")
        return df
    
    def cap_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cap outliers using IQR method."""
        if not self.cleaning_rules.get('cap_outliers', False):
            logger.info("✓ Outlier capping disabled in config")
            return df
        
        iqr_multiplier = self.cleaning_rules.get('iqr_multiplier', 1.5)
        
        for col in df.select_dtypes(include=['number']).columns:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - iqr_multiplier * iqr
            upper = q3 + iqr_multiplier * iqr
            
            outliers = ((df[col] < lower) | (df[col] > upper)).sum()
            if outliers > 0:
                df[col] = df[col].clip(lower, upper)
                logger.info(f"✓ Capped {outliers} outliers in '{col}'")
        
        return df
    
    def enforce_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enforce data types."""
        numeric_cols = df.select_dtypes(include=['number']).columns
        df[numeric_cols] = df[numeric_cols].astype(float)
        logger.info(f"✓ Enforced float type on {len(numeric_cols)} columns")
        return df
    
    def save_cleaned(self, df: pd.DataFrame, source: str = "phonopy") -> Path:
        """Save cleaned data."""
        source_config = self.config['data_sources'][source]
        base_name = source_config['filename'].replace('.csv', '_cleaned.csv')
        output_path = self.processed_data_dir / base_name
        
        df.to_csv(output_path, index=False)
        logger.info(f"✓ Saved: {output_path}")
        return output_path
    
    def clean_data(self, source: str = "phonopy") -> Path:
        """Run full cleaning pipeline."""
        df = self.load_data(source)
        logger.info(f"Original: {df.shape[0]} rows × {df.shape[1]} columns")
        
        df = self.handle_missing(df)
        df = self.remove_duplicates(df)
        df = self.cap_outliers(df)
        df = self.enforce_types(df)
        
        logger.info(f"Cleaned: {df.shape[0]} rows × {df.shape[1]} columns")
        output_path = self.save_cleaned(df, source)
        
        logger.info("="*50)
        logger.info("✓ Data cleaning completed")
        logger.info("="*50)
        
        return output_path


def main():
    """Run data cleaning pipeline."""
    try:
        cleaner = DataCleaning()
        cleaner.clean_data()
        return 0
    except Exception as e:
        logger.error(f"✗ Data cleaning failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
