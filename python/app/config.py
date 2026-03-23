import os
from pydantic_settings import BaseSettings
from pydantic import ConfigDict, Field
from typing import Optional

class IsolationForestConfig(BaseSettings):
    """Configuration for Isolation Forest anomaly detection"""

    # Expected proportion of anomalies in the dataset
    contamination: float = Field(default=0.1, ge=0.0, le=0.5,
                                description="Expected proportion of anomalies (0.0-0.5)")

    # Number of trees in the forest
    n_estimators: int = Field(default=100, ge=1, le=1000,
                             description="Number of trees in the forest")

    # Number of samples to draw for each tree
    max_samples: str = Field(default="auto",
                            description="Number of samples per tree ('auto' or float 0.0-1.0)")

    # Random state for reproducibility
    random_state: int = Field(default=42,
                             description="Random state for reproducibility")

    # Maximum features to consider for each tree
    max_features: float = Field(default=1.0, ge=0.1, le=1.0,
                               description="Max features per tree (0.1-1.0)")

    model_config = ConfigDict(env_prefix="ISOLATION_FOREST_")

class Settings(BaseSettings):
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")

    # Isolation Forest configuration
    isolation_forest: IsolationForestConfig = IsolationForestConfig()

    model_config = ConfigDict(env_file="../.env")

settings = Settings()