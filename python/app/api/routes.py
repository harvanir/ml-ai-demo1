from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services import anomaly_detector, ai_explainer
from app.services.ml_patterns import analyze_with_pattern, get_available_patterns
from app.utils.file_reader import read_file_from_buffer
from app.utils.dataframe_helper import get_data_summary
from app.models.response import AnalyzeResponse
from typing import Optional
import io

router = APIRouter()

@router.get("/health")
async def health_check():
    return {"status": "healthy"}

@router.get("/ml-patterns")
async def get_ml_patterns():
    """Get available ML learning patterns"""
    return {"patterns": get_available_patterns()}

@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_file(
    file: UploadFile = File(...),
    method: str = "iqr",
    use_ai: bool = False,
    # Isolation Forest parameters
    contamination: Optional[float] = None,
    n_estimators: Optional[int] = None,
    max_samples: Optional[float] = None,
    random_state: Optional[int] = None,
    max_features: Optional[float] = None
):
    """
    Analyze uploaded file for data summary and anomalies.

    Parameters:
    - method: Anomaly detection method ('iqr', 'zscore', 'isolation_forest')
    - use_ai: Whether to include AI explanation
    - contamination: Expected anomaly proportion (0.0-0.5) for isolation_forest
    - n_estimators: Number of trees for isolation_forest
    - max_samples: Samples per tree for isolation_forest
    - random_state: Random seed for isolation_forest
    - max_features: Max features per tree for isolation_forest
    """
    try:
        # Validate method
        valid_methods = ['iqr', 'zscore', 'isolation_forest']
        if method not in valid_methods:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid method. Choose from: {', '.join(valid_methods)}"
            )

        # Read file content
        content = await file.read()
        file_like = io.BytesIO(content)

        # Determine file type and read
        if file.filename.endswith('.csv'):
            df = read_file_from_buffer(file_like, 'csv')
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = read_file_from_buffer(file_like, 'excel')
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Use CSV or Excel.")

        if df is None or df.empty:
            raise HTTPException(status_code=400, detail="Empty or invalid file.")

        # Get data summary
        summary = get_data_summary(df)

        # Detect anomalies based on method
        if method == 'iqr':
            anomalies = anomaly_detector.detect_anomalies_iqr(df)
        elif method == 'zscore':
            anomalies = anomaly_detector.detect_anomalies_zscore(df)
        elif method == 'isolation_forest':
            anomalies = anomaly_detector.detect_anomalies_isolation_forest(
                df=df,
                contamination=contamination,
                n_estimators=n_estimators,
                max_samples=max_samples,
                random_state=random_state,
                max_features=max_features
            )
        else:
            anomalies = anomaly_detector.detect_anomalies(df)  # fallback

        # Generate AI explanation (always provided for demo)
        ai_summary = ai_explainer.explain_anomalies(anomalies)

        return AnalyzeResponse(
            summary=summary,
            anomalies=anomalies,
            ai_summary=ai_summary
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.post("/analyze-pattern")
async def analyze_with_ml_pattern(
    file: UploadFile = File(...),
    pattern: str = "supervised",
    method: str = "iqr"
):
    """
    Analyze uploaded file using specified ML learning pattern and anomaly detection method.

    Parameters:
    - pattern: ML learning pattern ('supervised', 'unsupervised', 'semi_supervised', 'reinforcement', 'deep_learning', 'ensemble')
    - method: Anomaly detection method ('iqr', 'zscore', 'isolation_forest')
    """
    try:
        # Validate pattern
        available_patterns = get_available_patterns()
        if pattern not in available_patterns:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid pattern. Available: {', '.join(available_patterns)}"
            )

        # Validate method
        valid_methods = ['iqr', 'zscore', 'isolation_forest']
        if method not in valid_methods:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid method. Choose from: {', '.join(valid_methods)}"
            )

        # Read file content
        content = await file.read()
        file_like = io.BytesIO(content)

        # Determine file type and read
        if file.filename.endswith('.csv'):
            df = read_file_from_buffer(file_like, 'csv')
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = read_file_from_buffer(file_like, 'excel')
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Use CSV or Excel.")

        if df is None or df.empty:
            raise HTTPException(status_code=400, detail="Empty or invalid file.")

        # Analyze with ML pattern
        pattern_result = analyze_with_pattern(df, pattern)

        # Get basic data summary
        summary = get_data_summary(df)

        return {
            "summary": summary,
            "ml_pattern": pattern_result,
            "pattern_used": pattern,
            "anomaly_method": method
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ML pattern analysis failed: {str(e)}")