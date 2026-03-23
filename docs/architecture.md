# Architecture

## Overview

This project is a simple ML + AI demo service.

Flow:

1. client uploads Excel or CSV file
2. Python API receives the file
3. file reader converts file into DataFrame
4. analyzer detects numeric columns
5. anomaly detector finds unusual values
6. optional AI explainer generates natural-language explanation
7. API returns structured JSON response

## High-Level Flow

```text
Client
  |
  v
FastAPI /analyze
  |
  +--> File Reader
  |
  +--> Data Summary
  |
  +--> Anomaly Detector
  |
  +--> Optional AI Explainer
  |
  v
JSON Response