# Predictive Maintenance System

An AI-powered Predictive Maintenance System that uses machine learning to estimate equipment failure risk from sensor data. The project combines a Random Forest model, a FastAPI backend, and a Streamlit dashboard to demonstrate industrial equipment monitoring and predictive maintenance.

## Overview

Unexpected equipment failures can cause production downtime, maintenance costs, and operational disruptions. This project demonstrates how machine learning can analyze sensor readings to identify equipment that may be at risk of failure.

The system uses the AI4I 2020 Predictive Maintenance Dataset to train and evaluate a classification model. Simulated sensor readings are used to demonstrate predictions and monitoring without requiring physical sensors or industrial hardware.

## Key Features

- **Machine Learning Predictions:** Estimates equipment failure risk using a trained Random Forest classifier.
- **Sensor Monitoring:** Processes temperature, rotational speed, torque, and tool-wear readings.
- **REST API:** Provides prediction and monitoring endpoints through FastAPI.
- **Interactive Dashboard:** Displays prediction statistics, risk trends, and historical records using Streamlit.
- **Failure Alerts:** Highlights predictions classified as potential failures.
- **Prediction History:** Stores prediction records in a local SQLite database.
- **Data Visualization:** Includes sensor distributions, correlation analysis, failure distribution, and a confusion matrix.
- **CSV Export:** Supports exporting prediction history for further analysis.

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Machine Learning | Scikit-learn, Random Forest |
| Backend API | FastAPI |
| API Server | Uvicorn |
| Dashboard | Streamlit |
| Data Processing | Pandas, NumPy |
| Model Storage | Joblib |
| Database | SQLite |
| Visualization | Matplotlib, Seaborn |

## Project Structure

```text
predictive-maintenance/
├── analysis_results/
│   ├── confusion_matrix.png
│   ├── correlation_heatmap.png
│   ├── evaluation_results.json
│   ├── failure_distribution.png
│   └── sensor_distributions.png
├── backend/
│   └── main.py
├── data/
│   └── ai4i2020.csv
├── models/
│   └── predictive_maintenance_model.joblib
├── frontend/
├── .gitignore
├── analyze_data.py
├── app.py
├── inspect_data.py
├── monitor.py
├── train_model.py
└── README.md
```

The `frontend/` directory is reserved for future development; the current interactive dashboard is implemented in `monitor.py`.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/arghadipsamanta234/predictive-maintenance-system.git
cd predictive-maintenance-system
```

### 2. Create a virtual environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can use the virtual environment's Python executable directly.

### 3. Install dependencies

```bash
python -m pip install pandas numpy scikit-learn joblib matplotlib seaborn fastapi uvicorn streamlit plotly
```

### 4. Train the machine learning model (optional)

The trained model is included in the repository. To retrain it using the dataset, run:

```bash
python train_model.py
```

Retraining may replace the existing model file.

## Running the Application

Open separate terminals in the project directory and activate the virtual environment in each terminal.

### Start the FastAPI backend

```bash
python -m uvicorn backend.main:app --reload
```

API documentation:

http://127.0.0.1:8000/docs

Health check:

http://127.0.0.1:8000/health

### Start the Streamlit dashboard

```bash
python -m streamlit run monitor.py
```

Open the dashboard at:

http://localhost:8501

Keep both processes running while testing features that depend on the API.

## API Endpoints

The backend provides endpoints for the following operations:

| Endpoint | Purpose |
|---|---|
| `GET /` | API information |
| `GET /health` | Service and dependency health |
| `POST /predict` | Predict equipment failure risk |
| `GET /history` | Retrieve historical predictions |
| `GET /alerts` | Retrieve potential failure alerts |
| `GET /statistics` | Retrieve prediction statistics |

Use the interactive Swagger documentation at `/docs` to inspect request schemas and test the endpoints.

## Model Evaluation

The Random Forest model was evaluated using a stratified train-test split on the AI4I 2020 dataset.

Recorded evaluation results:

| Metric | Result |
|---|---:|
| Accuracy | 97.85% |
| Balanced Accuracy | 82.57% |
| Failure Precision | 69% |
| Failure Recall | 66% |
| Failure F1-score | 0.68 |
| Failure Average Precision | 75.56% |

These results are from the project's evaluation split and are not a guarantee of performance on real industrial equipment. Accuracy alone can be misleading when equipment failures are relatively rare, so failure-class precision and recall are also reported.

## Dataset

This project uses the **AI4I 2020 Predictive Maintenance Dataset**, which contains synthetic industrial process data designed for predictive maintenance research.

The dataset includes variables such as:

- Air temperature
- Process temperature
- Rotational speed
- Torque
- Tool wear
- Machine failure labels

The included CSV file is `data/ai4i2020.csv`. Refer to the dataset provider's terms before redistributing the dataset separately.

## Limitations

- Sensor readings can be simulated; no physical sensors or connected industrial machines are required.
- Predictions indicate estimated risk based on the trained model, not confirmed equipment failures.
- Model performance on the AI4I dataset may differ from performance on real-world equipment.
- The system is an educational prototype and has not been validated for safety-critical industrial deployment.

## Future Improvements

- Integrate real-time sensor streams and IoT devices.
- Add authentication and role-based access control.
- Improve failure detection and probability calibration.
- Add configurable alert thresholds and notification services.
- Deploy the backend and dashboard to a cloud environment.
- Add automated testing, monitoring, and model-version management.

## Author

**Arghadip Samanta**

GitHub: [arghadipsamanta234](https://github.com/arghadipsamanta234)

## License

No license has been specified for this repository. Unless a license is added, others generally do not have permission to reuse, modify, or distribute the project's original code.