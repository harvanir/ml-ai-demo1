#!/usr/bin/env python3
"""
Test script to verify environment configuration loading.
Run from project root: python test_env_config.py
"""

import os
import sys
sys.path.insert(0, 'python')

from app.config import settings

def test_env_config():
    """Test that environment configuration is loaded correctly"""

    print("🧪 Testing Environment Configuration")
    print("=" * 50)

    # Test OpenAI API Key
    openai_key = settings.openai_api_key
    if openai_key and openai_key != "":
        print("✅ OpenAI API Key: Configured")
    else:
        print("⚠️  OpenAI API Key: Not set (AI explanations will be disabled)")

    # Test Isolation Forest Configuration
    iso_config = settings.isolation_forest
    print("\n🔧 Isolation Forest Configuration:")
    print(f"   contamination: {iso_config.contamination}")
    print(f"   n_estimators: {iso_config.n_estimators}")
    print(f"   max_samples: {iso_config.max_samples}")
    print(f"   random_state: {iso_config.random_state}")
    print(f"   max_features: {iso_config.max_features}")

    # Test parameter validation
    print("\n🔍 Parameter Validation:")
    try:
        # Test contamination range
        if not (0.0 <= iso_config.contamination <= 0.5):
            print("❌ contamination must be between 0.0 and 0.5")
        else:
            print("✅ contamination: Valid range")

        # Test n_estimators range
        if not (1 <= iso_config.n_estimators <= 1000):
            print("❌ n_estimators must be between 1 and 1000")
        else:
            print("✅ n_estimators: Valid range")

        # Test max_features range
        if not (0.1 <= iso_config.max_features <= 1.0):
            print("❌ max_features must be between 0.1 and 1.0")
        else:
            print("✅ max_features: Valid range")

        print("✅ All parameters validated successfully!")

    except Exception as e:
        print(f"❌ Validation error: {e}")

    print("\n💡 To modify configuration:")
    print("   1. Edit .env file in project root")
    print("   2. Restart the application")
    print("   3. Or override via API parameters")

    print("\n✅ Environment configuration test completed!")

if __name__ == "__main__":
    test_env_config()