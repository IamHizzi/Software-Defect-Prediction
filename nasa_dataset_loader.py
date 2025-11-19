"""
NASA Promise Dataset Loader
Loads and preprocesses NASA software defect datasets in ARFF format
"""

import numpy as np
import pandas as pd
import urllib.request
import os
from scipy.io import arff
import tempfile
import warnings
warnings.filterwarnings('ignore')


class NASADatasetLoader:
    """
    Loader for NASA Promise datasets
    Supports CM1, JM1, KC1, KC2, PC1, and other NASA datasets
    """

    # Dataset URLs - Using ApoorvaKrisna repository and PROMISE repository as fallback
    DATASET_URLS = {
        'CM1': 'https://raw.githubusercontent.com/ApoorvaKrisna/NASA-promise-dataset-repository/main/cm1.arff',
        'JM1': 'https://raw.githubusercontent.com/ApoorvaKrisna/NASA-promise-dataset-repository/main/jm1.arff',
        'KC1': 'https://raw.githubusercontent.com/ApoorvaKrisna/NASA-promise-dataset-repository/main/kc1.arff',
        'KC2': 'https://raw.githubusercontent.com/ApoorvaKrisna/NASA-promise-dataset-repository/main/kc2.arff',
        'PC1': 'https://raw.githubusercontent.com/ApoorvaKrisna/NASA-promise-dataset-repository/main/pc1.arff',
    }

    # Fallback URLs from PROMISE repository
    FALLBACK_URLS = {
        'CM1': 'http://promise.site.uottawa.ca/SERepository/datasets/cm1.arff',
        'JM1': 'http://promise.site.uottawa.ca/SERepository/datasets/jm1.arff',
        'KC1': 'http://promise.site.uottawa.ca/SERepository/datasets/kc1.arff',
        'KC2': 'http://promise.site.uottawa.ca/SERepository/datasets/kc2.arff',
        'PC1': 'http://promise.site.uottawa.ca/SERepository/datasets/pc1.arff',
    }

    def __init__(self, cache_dir='./nasa_datasets'):
        """
        Initialize the dataset loader

        Args:
            cache_dir: Directory to cache downloaded datasets
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def download_dataset(self, dataset_name):
        """
        Download a NASA dataset

        Args:
            dataset_name: Name of the dataset (e.g., 'CM1', 'JM1')

        Returns:
            Path to the downloaded file
        """
        if dataset_name not in self.DATASET_URLS:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(self.DATASET_URLS.keys())}")

        cache_path = os.path.join(self.cache_dir, f'{dataset_name}.arff')

        # Return if already cached
        if os.path.exists(cache_path):
            print(f"Using cached {dataset_name} from {cache_path}")
            return cache_path

        # Try primary URL first
        url = self.DATASET_URLS[dataset_name]
        print(f"Downloading {dataset_name} from {url}...")

        try:
            urllib.request.urlretrieve(url, cache_path)
            print(f"  Downloaded to {cache_path}")
            return cache_path
        except Exception as e:
            print(f"  Primary download failed: {e}")

            # Try fallback URL if available
            if dataset_name in self.FALLBACK_URLS:
                fallback_url = self.FALLBACK_URLS[dataset_name]
                print(f"  Trying fallback URL: {fallback_url}...")
                try:
                    urllib.request.urlretrieve(fallback_url, cache_path)
                    print(f"  Downloaded to {cache_path}")
                    return cache_path
                except Exception as e2:
                    print(f"  Fallback download also failed: {e2}")
                    raise e2
            else:
                raise e

    def load_arff(self, file_path):
        """
        Load ARFF file and convert to DataFrame

        Args:
            file_path: Path to ARFF file

        Returns:
            DataFrame with features and labels
        """
        try:
            # Load ARFF file
            data, meta = arff.loadarff(file_path)

            # Convert to DataFrame
            df = pd.DataFrame(data)

            # Decode byte strings if present
            for col in df.columns:
                if df[col].dtype == 'object':
                    try:
                        df[col] = df[col].str.decode('utf-8')
                    except:
                        pass

            return df

        except Exception as e:
            print(f"Error loading ARFF file {file_path}: {e}")
            raise

    def preprocess_dataset(self, df, target_column='defects'):
        """
        Preprocess NASA dataset

        Args:
            df: DataFrame with raw data
            target_column: Name of target column (default: 'defects')

        Returns:
            X (features), y (labels)
        """
        # Identify target column (common names in NASA datasets)
        possible_target_names = ['defects', 'Defective', 'defective', 'class', 'Class', 'bug']

        target_col = None
        for col in possible_target_names:
            if col in df.columns:
                target_col = col
                break

        if target_col is None:
            # Assume last column is target
            target_col = df.columns[-1]
            print(f"  Warning: Target column not found. Using last column: {target_col}")

        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Convert target to binary (0 = non-defective, 1 = defective)
        if y.dtype == 'object':
            # Handle string labels
            unique_vals = y.unique()
            if len(unique_vals) == 2:
                # Map to 0 and 1
                y = y.map({unique_vals[0]: 0, unique_vals[1]: 1})
            else:
                # Convert to numeric and binarize
                y = pd.to_numeric(y, errors='coerce')
                y = (y > 0).astype(int)
        else:
            # Already numeric
            if y.dtype == bool:
                y = y.astype(int)
            else:
                y = (y > 0).astype(int)

        # Handle missing values
        X = X.apply(pd.to_numeric, errors='coerce')
        X = X.fillna(X.median())

        # Remove non-numeric columns if any
        X = X.select_dtypes(include=[np.number])

        # Convert to numpy arrays
        X = X.values
        y = y.values

        return X, y

    def load_dataset(self, dataset_name):
        """
        Load and preprocess a NASA dataset

        Args:
            dataset_name: Name of the dataset (e.g., 'CM1', 'JM1')

        Returns:
            X (features), y (labels), dataset_info (metadata)
        """
        print(f"\n{'='*60}")
        print(f"Loading NASA Dataset: {dataset_name}")
        print('='*60)

        # Download dataset
        file_path = self.download_dataset(dataset_name)

        # Load ARFF file
        df = self.load_arff(file_path)
        print(f"  Raw data shape: {df.shape}")

        # Preprocess
        X, y = self.preprocess_dataset(df)
        print(f"  Processed shape: X={X.shape}, y={y.shape}")

        # Dataset info
        dataset_info = {
            'name': dataset_name,
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'n_defective': int(np.sum(y)),
            'n_clean': int(np.sum(1 - y)),
            'defect_rate': float(np.mean(y)),
            'feature_names': df.columns.tolist()[:-1] if len(df.columns) > 0 else None
        }

        print(f"  Total samples: {dataset_info['n_samples']}")
        print(f"  Features: {dataset_info['n_features']}")
        print(f"  Defective: {dataset_info['n_defective']} ({dataset_info['defect_rate']*100:.1f}%)")
        print(f"  Clean: {dataset_info['n_clean']} ({(1-dataset_info['defect_rate'])*100:.1f}%)")

        return X, y, dataset_info

    def load_all_datasets(self, dataset_names=None):
        """
        Load multiple NASA datasets

        Args:
            dataset_names: List of dataset names to load (default: all available)

        Returns:
            Dictionary mapping dataset names to (X, y, info) tuples
        """
        if dataset_names is None:
            dataset_names = ['CM1', 'JM1', 'KC1', 'KC2', 'PC1']

        datasets = {}

        print(f"\n{'='*60}")
        print(f"Loading {len(dataset_names)} NASA Datasets")
        print('='*60)

        for name in dataset_names:
            try:
                X, y, info = self.load_dataset(name)
                datasets[name] = (X, y, info)
            except Exception as e:
                print(f"  Failed to load {name}: {e}")
                continue

        print(f"\n{'='*60}")
        print(f"Successfully loaded {len(datasets)} datasets")
        print('='*60)

        return datasets


def get_dataset_summary(datasets):
    """
    Generate summary statistics for loaded datasets

    Args:
        datasets: Dictionary of loaded datasets

    Returns:
        DataFrame with summary statistics
    """
    summary_data = []

    for name, (X, y, info) in datasets.items():
        summary_data.append({
            'Dataset': name,
            'Samples': info['n_samples'],
            'Features': info['n_features'],
            'Defective': info['n_defective'],
            'Clean': info['n_clean'],
            'Defect Rate (%)': f"{info['defect_rate']*100:.2f}"
        })

    summary_df = pd.DataFrame(summary_data)
    return summary_df


if __name__ == "__main__":
    # Test the loader
    loader = NASADatasetLoader()

    # Load specific datasets
    datasets = loader.load_all_datasets(['CM1', 'JM1', 'KC1', 'KC2', 'PC1'])

    # Print summary
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    summary = get_dataset_summary(datasets)
    print(summary.to_string(index=False))
