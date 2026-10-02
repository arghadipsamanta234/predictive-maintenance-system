
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load dataset
data_path = Path("data") / "ai4i2020.csv"
df = pd.read_csv(data_path)

# Create a folder for charts
output_dir = Path("analysis_results")
output_dir.mkdir(exist_ok=True)

# 1. Machine failure distribution
failure_counts = df["Machine failure"].value_counts().sort_index()

plt.figure(figsize=(7, 5))
sns.barplot(
    x=failure_counts.index,
    y=failure_counts.values
)
plt.title("Machine Failure Distribution")
plt.xlabel("Machine Failure (0 = No, 1 = Yes)")
plt.ylabel("Number of Records")
plt.tight_layout()
plt.savefig(output_dir / "failure_distribution.png")
plt.show()

# 2. Sensor and operating measurements
features = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]"
]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for ax, feature in zip(axes, features):
    sns.histplot(data=df, x=feature, hue="Machine failure",
                 bins=30, element="step", stat="count",
                 common_norm=False, ax=ax)
    ax.set_title(feature)

plt.suptitle("Operating Measurements by Machine Failure")
plt.tight_layout()
plt.savefig(output_dir / "sensor_distributions.png")
plt.show()

# 3. Correlation heatmap
numeric_df = df.select_dtypes(include="number")
correlation = numeric_df.corr()

plt.figure(figsize=(12, 9))
sns.heatmap(correlation, cmap="coolwarm", center=0)
plt.title("Correlation Between Numerical Features")
plt.tight_layout()
plt.savefig(output_dir / "correlation_heatmap.png")
plt.show()

print("\nAnalysis completed!")
print("Charts saved in:", output_dir.resolve())
print("\nFailure counts:")
print(failure_counts)
print("\nFailure percentage:")
print((failure_counts / len(df) * 100).round(2))
