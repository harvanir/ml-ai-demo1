# Project Name
ml-ai-demo1

## Vision
ml-ai-demo1 is a foundational demo project for building a practical ML + AI file analysis system.

The short-term goal is to create a simple working MVP in Python.
The longer-term goal is to evolve this into a more modular and extensible system that can later include Golang services and more advanced analysis capabilities.

This project should be treated as a staged evolution, not as a one-off script.

---

## Strategic Direction

### Phase 1: Python-first MVP
Build the first working implementation in Python because Python is the fastest and most practical language for:
- data analysis
- anomaly detection
- machine learning experimentation
- LLM integration

The first version should remain intentionally simple.

### Phase 2: Stable contracts and modularization
As the Python implementation becomes stable, preserve response contracts and modular boundaries so other components can later integrate without major rewrites.

### Phase 3: Golang integration
Introduce Golang as a separate implementation area for purposes such as:
- API gateway
- orchestration
- service wrapper
- production-friendly backend integration

At this stage, Golang does not need to duplicate all ML logic immediately.
It may initially call Python services over HTTP or consume shared response contracts.

### Phase 4: Extended capabilities
After the MVP is stable, the project may evolve into richer capabilities such as:
- multi-sheet analysis
- deeper column profiling
- richer anomaly detection
- AI-generated explanation and recommendations
- file-to-structure transformation
- reusable analysis pipeline patterns

---

## Architectural Philosophy

This repository should be designed to support growth without overengineering the MVP.

Key principles:
- start simple
- separate concerns early
- preserve contracts
- keep implementation modular
- allow Python and Golang to coexist
- avoid premature complexity

---

## Why the Repository Is Split Early

The repository is intentionally structured with separate directories for:
- `python/`
- `golang/`
- `shared/`

This is not because all layers are implemented now.
It is because the project is expected to evolve.

### `python/`
Main implementation for the current phase.
This is where ML and AI logic are first built.

### `golang/`
Reserved for future use.
This may later become:
- a wrapper service
- an orchestration layer
- an API layer
- a production-facing service

### `shared/`
Reserved for contracts, schemas, and shared examples.
This is important to keep interfaces stable as the project grows.

---

## MVP Scope vs Long-Term Direction

### MVP Scope Now
- upload a file
- read Excel or CSV
- summarize dataset
- detect numeric columns
- detect anomalies
- return JSON response
- optionally generate AI summary

### Long-Term Direction Later
- richer dataset profiling
- better anomaly detection methods
- more robust AI explanation
- more formal contracts
- Golang integration
- possible service-to-service communication
- possible support for transformation and mapping workflows

---

## What AI Coding Assistants Should Understand

When helping with this project, AI coding assistants should respect the following:

1. The current target is a working Python MVP.
2. The code should be written cleanly so it can later be wrapped or consumed by Golang.
3. Shared contracts should remain stable whenever possible.
4. Do not introduce unnecessary complexity too early.
5. Avoid building future phases now unless explicitly requested.
6. However, do preserve code organization so future phases are easy to add.

---

## Long-Term Possible Evolution

This project may later become one of the following patterns:

### Pattern A: Python analysis engine + Golang API wrapper
- Python handles data analysis and AI logic
- Golang handles routing, orchestration, or integration

### Pattern B: Python prototype, then selective Golang port
- Python proves logic first
- selected components are later reimplemented in Golang if needed

### Pattern C: Shared contracts, polyglot services
- Python and Golang coexist
- both rely on stable schemas from `shared/`

Do not force a final decision now.
The codebase should simply remain compatible with these directions.

---

## Future ML Evolution

Current anomaly detection can start with:
- IQR
- z-score

Possible future upgrades:
- Isolation Forest
- clustering-based detection
- forecasting
- richer statistical profiling

Current phase should prioritize explainability and speed of delivery over sophistication.

---

## Future AI Evolution

Current AI usage can remain optional and lightweight.

Possible future directions:
- anomaly explanation
- dataset summarization
- column semantic interpretation
- recommendation generation
- transformation guidance

Do not tightly couple the whole system to a specific AI provider too early.

---

## Final Reminder

This project is intentionally staged.

What matters now:
- build the MVP
- keep the structure clean
- keep the interfaces clear
- leave room for Golang later

Do not overbuild for the future, but do not ignore the future either.