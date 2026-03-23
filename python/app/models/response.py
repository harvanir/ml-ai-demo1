from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class DataSummary(BaseModel):
    total_rows: int
    total_columns: int
    numeric_columns: List[str]

class Anomaly(BaseModel):
    column: str
    value: float
    index: int
    z_score: Optional[float] = None
    is_outlier: bool

class AnalyzeResponse(BaseModel):
    summary: DataSummary
    anomalies: List[Anomaly]
    ai_summary: Optional[str] = None