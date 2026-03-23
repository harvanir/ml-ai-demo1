import pandas as pd
from typing import List

def get_numeric_columns(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=['number']).columns.tolist()

def get_data_summary(df: pd.DataFrame) -> dict:
    return {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "numeric_columns": get_numeric_columns(df)
    }