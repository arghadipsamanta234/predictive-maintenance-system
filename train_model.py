
from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "ai4i2020.csv"
MODEL_PATH = BASE_DIR / "models" / "predictive_maintenance_model.joblib"
RESULTS_DIR = BASE_DIR / "analysis_results"

TARGET = "Machine failure"

RANDOM_STATE = 42
TEST_SIZE = 0.20

# Used only for threshold comparison, not as the deployed threshold.
THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70]


# ============================================================
# 2. LOAD DATASET
# ============================================================

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Check that data/ai4i2020.csv exists."
    )

df = pd.read_csv(DATA_PATH)

if TARGET not in df.columns:
    raise ValueError(
        f"Target column '{TARGET}' is missing from the dataset."
    )

print("=" * 60)
print("PREDICTIVE MAINTENANCE - MODEL TRAINING")
print("=" * 60)
print(f"Dataset records: {len(df)}")
print(f"Dataset columns: {len(df.columns)}")


# ============================================================
# 3. SELECT FEATURES AND TARGET
# ============================================================

excluded_columns = [
    "UDI",
    "Product ID",
    TARGET,
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
]

X = df.drop(columns=excluded_columns, errors="ignore")
y = df[TARGET]

if y.nunique() != 2:
    raise ValueError(
        f"Expected a binary target, but found: {y.unique()}"
    )

if set(y.unique()) != {0, 1}:
    raise ValueError(
        "Expected target values 0 (no failure) and 1 (failure)."
    )

print("\nTarget distribution:")
print(y.value_counts().sort_index().rename({
    0: "No Failure",
    1: "Failure",
}))


# ============================================================
# 4. IDENTIFY FEATURE TYPES
# ============================================================

# The AI4I dataset uses "Type" as its categorical feature.
# Explicitly selecting it avoids pandas string-dtype warnings.
categorical_features = [
    column
    for column in ["Type"]
    if column in X.columns
]

numerical_features = [
    column
    for column in X.columns
    if column not in categorical_features
]

print("\nCategorical features:", categorical_features)
print("Numerical features:", numerical_features)


# ============================================================
# 5. PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features,
        ),
        (
            "numerical",
            "passthrough",
            numerical_features,
        ),
    ],
    remainder="drop",
)


# ============================================================
# 6. CREATE MODEL PIPELINE
# ============================================================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
    ]
)


# ============================================================
# 7. SPLIT DATA
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# ============================================================
# 8. TRAIN MODEL
# ============================================================

print("\nTraining model...")
model.fit(X_train, y_train)

# Default predictions use the estimator's standard threshold.
y_pred = model.predict(X_test)

# Explicitly locate class 1 rather than assuming its column position.
classifier = model.named_steps["classifier"]
failure_class_index = list(classifier.classes_).index(1)
y_prob = model.predict_proba(X_test)[:, failure_class_index]


# ============================================================
# 9. CLASSIFICATION METRICS
# ============================================================

accuracy = accuracy_score(y_test, y_pred)
balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
avg_precision = average_precision_score(y_test, y_prob)

report = classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["No Failure", "Failure"],
    output_dict=True,
    zero_division=0,
)

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)
print(f"Accuracy:            {accuracy:.4f}")
print(f"Balanced accuracy:   {balanced_accuracy:.4f}")
print(f"Failure avg precision: {avg_precision:.4f}")

print("\nCLASSIFICATION REPORT")
print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["No Failure", "Failure"],
        zero_division=0,
    )
)


# ============================================================
# 10. CONFUSION MATRIX WITH EXPLICIT COUNTS
# ============================================================

# Rows = actual class; columns = predicted class.
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

tn, fp, fn, tp = cm.ravel()

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)
print("Rows: actual class | Columns: predicted class")
print("                 Predicted Normal | Predicted Failure")
print(f"Actual Normal:   {tn:15d} | {fp:16d}")
print(f"Actual Failure:  {fn:15d} | {tp:16d}")

print("\nInterpretation:")
print(f"True negatives  (TN): {tn}")
print(f"False positives (FP): {fp}")
print(f"False negatives (FN): {fn}")
print(f"True positives  (TP): {tp}")
print(f"Actual failures detected: {tp} out of {tp + fn}")
print(f"Actual failures missed:   {fn}")

# Save the confusion matrix plot.
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Failure", "Failure"],
).plot(
    ax=ax,
    cmap="Blues",
    values_format="d",
    colorbar=False,
)
ax.set_title("Random Forest - Confusion Matrix")
fig.tight_layout()
fig.savefig(
    RESULTS_DIR / "confusion_matrix.png",
    dpi=150,
    bbox_inches="tight",
)
plt.close(fig)


# ============================================================
# 11. COMPARE FAILURE PROBABILITY THRESHOLDS
# ============================================================

print("\n" + "=" * 60)
print("FAILURE THRESHOLD COMPARISON")
print("=" * 60)
print(
    "Threshold | Precision | Recall | F1-score | "
    "False Alarms | Missed Failures"
)

threshold_results = []

for threshold in THRESHOLDS:
    threshold_pred = (y_prob >= threshold).astype(int)

    threshold_cm = confusion_matrix(
        y_test,
        threshold_pred,
        labels=[0, 1],
    )
    threshold_tn, threshold_fp, threshold_fn, threshold_tp = (
        threshold_cm.ravel()
    )

    precision = precision_score(
        y_test, threshold_pred, zero_division=0
    )
    recall = recall_score(
        y_test, threshold_pred, zero_division=0
    )
    f1 = f1_score(
        y_test, threshold_pred, zero_division=0
    )

    print(
        f"{threshold:9.2f} | "
        f"{precision:9.3f} | "
        f"{recall:6.3f} | "
        f"{f1:8.3f} | "
        f"{threshold_fp:12d} | "
        f"{threshold_fn:15d}"
    )

    threshold_results.append({
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "true_negatives": int(threshold_tn),
        "false_positives": int(threshold_fp),
        "false_negatives": int(threshold_fn),
        "true_positives": int(threshold_tp),
    })


# ============================================================
# 12. SAVE EVALUATION RESULTS
# ============================================================

summary = {
    "dataset_records": int(len(df)),
    "training_records": int(len(X_train)),
    "testing_records": int(len(X_test)),
    "accuracy": float(accuracy),
    "balanced_accuracy": float(balanced_accuracy),
    "failure_average_precision": float(avg_precision),
    "confusion_matrix": {
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    },
    "classification_report": report,
    "threshold_comparison": threshold_results,
}

with open(
    RESULTS_DIR / "evaluation_results.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(summary, file, indent=4)


# ============================================================
# 13. SAVE TRAINED MODEL
# ============================================================

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model, MODEL_PATH)

print("\n" + "=" * 60)
print("TRAINING COMPLETED")
print("=" * 60)
print(f"Model saved: {MODEL_PATH}")
print(f"Confusion matrix image: {RESULTS_DIR / 'confusion_matrix.png'}")
print(f"Evaluation results: {RESULTS_DIR / 'evaluation_results.json'}")
print("\nThe deployed model still uses its default prediction threshold.")
print("Threshold comparisons above are evaluation only.")
