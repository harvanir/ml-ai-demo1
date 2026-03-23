import pandas as pd
from typing import Optional, BinaryIO

def read_file(file_path: str) -> Optional[pd.DataFrame]:
    """
    Read CSV or Excel file and return DataFrame.
    For Excel, reads the first sheet.
    """
    try:
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            return pd.read_excel(file_path, sheet_name=0)
        else:
            raise ValueError("Unsupported file format. Use CSV or Excel.")
    except Exception as e:
        raise ValueError(f"Error reading file: {str(e)}")

def read_file_from_buffer(buffer: BinaryIO, file_type: str) -> Optional[pd.DataFrame]:
    """
    Read CSV or Excel from binary buffer.
    """
    try:
        if file_type == 'csv':
            return pd.read_csv(buffer)
        elif file_type == 'excel':
            return pd.read_excel(buffer, sheet_name=0)
        else:
            raise ValueError("Unsupported file type.")
    except Exception as e:
        raise ValueError(f"Error reading file: {str(e)}")