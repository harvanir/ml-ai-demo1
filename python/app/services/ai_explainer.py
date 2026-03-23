from typing import List, Optional
from app.models.response import Anomaly
from app.config import settings

def explain_anomalies(anomalies: List[Anomaly]) -> Optional[str]:
    """
    Generate a natural language explanation of anomalies using LLM.
    Provides basic AI-like summary even without API key for demo purposes.
    """
    if not anomalies:
        return "✅ No anomalies detected in the data. All values appear to be within normal ranges."

    # Group anomalies by column
    anomalies_by_column = {}
    for anomaly in anomalies:
        col = anomaly.column
        if col not in anomalies_by_column:
            anomalies_by_column[col] = []
        anomalies_by_column[col].append(anomaly)

    # Generate summary
    total_anomalies = len(anomalies)
    affected_columns = len(anomalies_by_column)

    summary_parts = []

    if settings.openai_api_key:
        # TODO: Use actual OpenAI API for more sophisticated explanations
        summary_parts.append(f"🤖 AI Analysis: Found {total_anomalies} anomalies across {affected_columns} columns.")
    else:
        # Mock AI summary for demo
        summary_parts.append(f"🤖 AI Analysis: Detected {total_anomalies} unusual values across {affected_columns} columns that fall outside normal statistical ranges.")

    # Add details about each column
    for col, col_anomalies in anomalies_by_column.items():
        count = len(col_anomalies)
        values = [f"{a.value:.2f}" for a in col_anomalies[:3]]  # Show first 3 values
        if count > 3:
            values.append("...")
        value_str = ", ".join(values)

        if count == 1:
            summary_parts.append(f"📊 Column '{col}': One outlier value ({value_str}) detected.")
        else:
            summary_parts.append(f"📊 Column '{col}': {count} outlier values detected including {value_str}.")

    # Add recommendations
    summary_parts.append("💡 Recommendation: Review these values for potential data entry errors, system issues, or legitimate but unusual business cases.")

    return " ".join(summary_parts)