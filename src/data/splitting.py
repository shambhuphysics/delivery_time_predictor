"""
Data Splitting Module
Config-driven train/val/test splitting following MLOps principles.
"""
import logging
import pandas as pd
import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataSplit:
    """Splits cleaned data into train/val/test sets."""
    
    def __init__(self, config_path: str = "configs/data.yaml"):
        self.config = self._load_config(config_path)
        self.project_root = Path.cwd()
        self.processed_dir = self.project_root / self.config['paths']['processed']
        self.split_config = self.config['splitting']
        self.seed = self.split_config['random_seed']
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_data(self, source: str = "phonopy") -> pd.DataFrame:
        """Load cleaned data."""
        source_config = self.config['data_sources'][source]
        base_name = source_config['filename'].replace('.csv', '_cleaned.csv')
        input_path = self.processed_dir / base_name
        
        logger.info(f"Loading: {input_path}")
        return pd.read_csv(input_path)
    
    def split_data(self, df: pd.DataFrame) -> tuple:
        """Split data into train/val/test sets."""
        test_size = self.split_config['test_size']
        val_size = self.split_config['val_size']
        shuffle = self.split_config['shuffle']
        
        # First split: separate test set
        df_train, df_test = train_test_split(
            df,
            test_size=test_size,
            random_state=self.seed,
            shuffle=shuffle
        )
        
        # Second split: separate validation from training
        val_ratio = val_size / (1 - test_size)
        df_train, df_val = train_test_split(
            df_train,
            test_size=val_ratio,
            random_state=self.seed,
            shuffle=shuffle
        )
        
        logger.info(f"✓ Split sizes - Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")
        logger.info(f"✓ Split ratios - Train: {len(df_train)/len(df):.2%} | Val: {len(df_val)/len(df):.2%} | Test: {len(df_test)/len(df):.2%}")
        
        return df_train, df_val, df_test
    
    def save_split(self, df: pd.DataFrame, split_name: str, source: str = "phonopy"):
        """Save split to CSV."""
        source_config = self.config['data_sources'][source]
        base_name = source_config['filename'].replace('.csv', f'_{split_name}.csv')
        output_path = self.processed_dir / base_name
        
        df.to_csv(output_path, index=False)
        logger.info(f"✓ Saved {split_name}: {output_path}")
    
    def run(self, source: str = "phonopy"):
        """Run complete splitting pipeline."""
        df = self.load_data(source)
        logger.info(f"Original data: {len(df)} rows × {len(df.columns)} columns")
        
        df_train, df_val, df_test = self.split_data(df)
        
        self.save_split(df_train, "train", source)
        self.save_split(df_val, "val", source)
        self.save_split(df_test, "test", source)
        
        logger.info("="*50)
        logger.info("✓ Data splitting completed")
        logger.info("="*50)


def main():
    """Run data splitting pipeline."""
    try:
        splitter = DataSplit()
        splitter.run()
        return 0
    except Exception as e:
        logger.error(f"✗ Data splitting failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
