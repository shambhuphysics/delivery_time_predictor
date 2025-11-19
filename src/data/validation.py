"""Data validation following MLOps best practices."""

import logging
import pandas as pd
import yaml
from pathlib import Path
from typing import Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataValidation:
    """Validates data schema and quality."""
    
    def __init__(self, config_path: str = "configs/data.yaml"):
        self.config = self._load_config(config_path)
        self.project_root = Path.cwd()
        self.raw_data_dir = self.project_root / self.config['paths']['raw']
        self.validation_rules = self.config['validation']
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def validate_missing_values(self, df: pd.DataFrame) -> bool:
        """Check missing value threshold."""
        threshold = self.validation_rules['missing_value_threshold']
        missing_pct = df.isnull().sum() / len(df)
        high_missing = missing_pct[missing_pct > threshold]
        
        if not high_missing.empty:
            logger.warning(f"Columns with >{threshold*100}% missing: {list(high_missing.index)}")
        
        total_missing = df.isnull().sum().sum()
        logger.info(f"✓ Missing values: {total_missing} ({total_missing/df.size*100:.2f}%)")
        return True
    
    def validate_duplicates(self, df: pd.DataFrame) -> bool:
        """Check for duplicate rows."""
        n_duplicates = df.duplicated().sum()
        
        if n_duplicates > 0:
            if not self.validation_rules['allow_duplicates']:
                raise ValueError(f"Found {n_duplicates} duplicate rows (not allowed)")
            logger.warning(f"Found {n_duplicates} duplicate rows")
        else:
            logger.info("✓ No duplicate rows")
        return True
    
    def validate_value_ranges(self, df: pd.DataFrame) -> bool:
        """Validate numeric value ranges."""
        numeric_df = df.select_dtypes(include=['number'])
        
        for col in numeric_df.columns:
            # Check infinite values
            if numeric_df[col].isin([float('inf'), float('-inf')]).any():
                if not self.validation_rules['allow_infinite_values']:
                    raise ValueError(f"Column '{col}' contains infinite values")
            
            # Check negative values
            if (numeric_df[col] < 0).any():
                if not self.validation_rules['allow_negative_values']:
                    raise ValueError(f"Column '{col}' contains negative values")
        
        logger.info(f"✓ Value ranges checked: {len(numeric_df.columns)} numeric columns")
        return True
    
    def validate_shape(self, df: pd.DataFrame) -> bool:
        """Validate dataset shape."""
        min_rows = self.validation_rules['min_rows']
        min_cols = self.validation_rules['min_columns']
        
        if len(df) < min_rows:
            raise ValueError(f"Dataset has {len(df)} rows, minimum {min_rows} required")
        if len(df.columns) < min_cols:
            raise ValueError(f"Dataset has {len(df.columns)} columns, minimum {min_cols} required")
        
        logger.info(f"✓ Shape valid: {df.shape[0]:,} rows × {df.shape[1]} columns")
        return True
    
    def validate_data(self, source: str = "phonopy") -> Dict:
        """Run all validation checks."""
        source_config = self.config['data_sources'][source]
        data_path = self.raw_data_dir / source_config['filename']
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        logger.info(f"Validating: {data_path}")
        df = pd.read_csv(data_path)
        
        # Run checks
        self.validate_shape(df)
        self.validate_missing_values(df)
        self.validate_duplicates(df)
        self.validate_value_ranges(df)
        
        # Summary
        stats = {
            'rows': len(df),
            'columns': len(df.columns),
            'memory_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'missing_total': df.isnull().sum().sum(),
            'duplicates': df.duplicated().sum()
        }
        
        logger.info("="*50)
        logger.info("✓ Validation passed!")
        logger.info(f"  Rows: {stats['rows']:,}")
        logger.info(f"  Columns: {stats['columns']}")
        logger.info(f"  Memory: {stats['memory_mb']:.2f} MB")
        logger.info("="*50)
        
        return stats


def main():
    """Run data validation pipeline."""
    try:
        validation = DataValidation()
        validation.validate_data()
        logger.info("✓ Data validation completed")
        return 0
    except Exception as e:
        logger.error(f"✗ Data validation failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
