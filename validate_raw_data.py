# src/data_validation/validate_raw_data.py
import pandas as pd
from pathlib import Path


def validate_data(file_path):
    df = pd.read_csv(file_path)

    # Basic checks
    assert not df.empty, "Empty DataFrame"
    assert df.isna().sum().sum() == 0, "Null values detected"

    if 'timestamp' in df.columns:
        assert pd.api.types.is_datetime64_any_dtype(df['timestamp']), "Invalid timestamps"

    print(f"Validation passed for {file_path.name}")
    return True


# Validate all raw files
raw_files = Path('data/raw').glob('*.csv')
[validate_data(f) for f in raw_files]
