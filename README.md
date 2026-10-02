# Predictive Maintenance System

## Overview
A machine learning-based predictive maintenance prototype that analyzes simulated machine sensor readings to estimate the probability of equipment failure.

## Features
- Simulated machine sensor data
- Random Forest classification model
- Failure probability estimation
- FastAPI REST API
- SQLite prediction history
- Streamlit monitoring dashboard
- Failure alert history
- Sensor trend visualization
- CSV export of prediction records

## Technology Stack
- Python
- Pandas and NumPy
- Scikit-learn
- FastAPI
- Streamlit
- SQLite
- Matplotlib

## System Workflow
1. Generate simulated machine sensor readings.
2. Send readings to the FastAPI backend.
3. Process the readings with the trained ML model.
4. Calculate the estimated failure probability.
5. Store prediction results in SQLite.
6. Display machine readings, trends, and alerts on the Streamlit dashboard.

## Running the Project

Start the API from the project root:

```bash
python -m uvicorn backend.main:app --reload
```

In a second terminal, start the dashboard:

```bash
python -m streamlit run monitor.py
```

API documentation: http://127.0.0.1:8000/docs

Dashboard: http://localhost:8501

## Limitations
This is an experimental prototype using simulated sensor readings and the AI4I 2020 predictive maintenance dataset. Predictions have not been validated for real industrial safety decisions. The system should not be used as the sole basis for maintenance or safety actions.