
# Implementation Plan

## Phase 1
Set up project structure and FastAPI app

Deliverables:
- folder structure
- requirements.txt
- main.py
- health check endpoint

## Phase 2
Implement file reading

Deliverables:
- read CSV
- read Excel first sheet
- return DataFrame
- basic validation

## Phase 3
Implement summary and numeric column detection

Deliverables:
- total rows
- total columns
- numeric columns list

## Phase 4
Implement anomaly detection

Deliverables:
- IQR or z-score anomaly detection
- anomaly list output

## Phase 5
Implement `/analyze` API

Deliverables:
- upload file
- process file
- return JSON

## Phase 6
Optional AI explanation

Deliverables:
- LLM summarization for anomaly results
- `ai_summary` response field