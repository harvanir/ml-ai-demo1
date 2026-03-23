"""
Machine Learning Patterns Demo
This module demonstrates different ML learning paradigms
"""

from enum import Enum
from typing import Dict, Any, List, Optional
import pandas as pd
from app.models.response import AnalyzeResponse, DataSummary

class MLPattern(Enum):
    SUPERVISED = "supervised"
    UNSUPERVISED = "unsupervised"
    SEMI_SUPERVISED = "semi_supervised"
    REINFORCEMENT = "reinforcement"
    DEEP_LEARNING = "deep_learning"
    ENSEMBLE = "ensemble"

def get_available_patterns() -> List[str]:
    """Get list of available ML patterns"""
    return [pattern.value for pattern in MLPattern]

def analyze_with_pattern(
    df: pd.DataFrame,
    pattern: str,
    **kwargs
) -> Dict[str, Any]:
    """
    Analyze data using specified ML pattern

    Args:
        df: Input DataFrame
        pattern: ML pattern to use
        **kwargs: Additional parameters for the pattern

    Returns:
        Dictionary with analysis results
    """
    pattern_enum = MLPattern(pattern)

    if pattern_enum == MLPattern.SUPERVISED:
        from .supervised import analyze_supervised
        return analyze_supervised(df, **kwargs)
    elif pattern_enum == MLPattern.UNSUPERVISED:
        from .unsupervised import analyze_unsupervised
        return analyze_unsupervised(df, **kwargs)
    elif pattern_enum == MLPattern.SEMI_SUPERVISED:
        from .semi_supervised import analyze_semi_supervised
        return analyze_semi_supervised(df, **kwargs)
    elif pattern_enum == MLPattern.REINFORCEMENT:
        from .reinforcement import analyze_reinforcement
        return analyze_reinforcement(df, **kwargs)
    elif pattern_enum == MLPattern.DEEP_LEARNING:
        from .deep_learning import analyze_deep_learning
        return analyze_deep_learning(df, **kwargs)
    elif pattern_enum == MLPattern.ENSEMBLE:
        from .ensemble import analyze_ensemble_learning
        return analyze_ensemble_learning(df, **kwargs)
    else:
        return {"error": f"Unknown pattern: {pattern}"}

# Import pattern implementations
from .supervised import analyze_supervised
from .unsupervised import analyze_unsupervised
from .semi_supervised import analyze_semi_supervised
from .reinforcement import analyze_reinforcement
from .deep_learning import analyze_deep_learning
from .ensemble import analyze_ensemble_learning