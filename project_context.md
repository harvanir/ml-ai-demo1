# Project Name
ml-ai-demo1

## Project Goal
This project is a simple demo for combining Machine Learning and AI in a practical file analysis workflow.

The system should:
1. accept an uploaded Excel or CSV file
2. read and summarize the data
3. detect anomalies in numeric columns using simple machine learning or statistical logic
4. optionally use an LLM to explain the anomaly results in human-readable language

This project is intentionally designed as a simple MVP first.
Do not overengineer it.

---

## Current Development Strategy

We are implementing the Python version first.

Why:
- Python is better for ML and AI experimentation
- it allows faster iteration for data processing and anomaly detection
- Golang will be added later as a separate directory

The repository should already be structured to support:
- `python/` as the main implementation now
- `golang/` as a future implementation later

---

## Monorepo Structure

```text
ml-ai-demo1/
├── README.md
├── .gitignore
├── docs/
│   ├── architecture.md
│   ├── api-contract.md
│   └── implementation-plan.md
├── sample_data/
│   └── sample.xlsx
├── python/
│   ├── app/
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── api/
│   │   │   └── routes.py
│   │   ├── services/
│   │   │   ├── file_reader.py
│   │   │   ├── anomaly_detector.py
│   │   │   └── ai_explainer.py
│   │   ├── models/
│   │   │   └── response.py
│   │   └── utils/
│   │       └── dataframe_helper.py
│   ├── tests/
│   │   └── test_anomaly_detector.py
│   ├── requirements.txt
│   └── .env.example
├── golang/
│   ├── cmd/
│   │   └── server/
│   │       └── main.go
│   ├── internal/
│   ├── go.mod
│   └── README.md
└── shared/
    ├── schemas/
    │   ├── analyze_request.json
    │   └── analyze_response.json
    └── examples/
        └── analyze_response_example.json