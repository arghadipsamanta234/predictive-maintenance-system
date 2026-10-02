import random
from datetime import datetime

import pandas as pd
import requests
import streamlit as st


# ============================================================
# 1. CONFIGURATION
# ============================================================

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Predictive Maintenance Monitor",
    page_icon="🏭",
    layout="wide",
)


# ============================================================
# 2. CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
    }

    .risk-box {
        padding: 15px;
        border-radius: 10px;
        margin-top: 10px;
        text-align: center;
        font-size: 1.2rem;
        font-weight: 600;
    }

    .small-text {
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. PAGE HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🏭 Predictive Maintenance Monitor</div>',
    unsafe_allow_html=True,
)

st.caption(
    "Real-time simulated machine monitoring connected to the "
    "ML prediction API with persistent SQLite history."
)


# ============================================================
# 4. SESSION STATE
# ============================================================

if "monitoring" not in st.session_state:
    st.session_state.monitoring = False

if "machine_state" not in st.session_state:
    st.session_state.machine_state = {
        "air": 298.1,
        "speed": 1500.0,
        "torque": 40.0,
        "wear": 100.0,
    }

if "latest_reading" not in st.session_state:
    st.session_state.latest_reading = None

if "latest_result" not in st.session_state:
    st.session_state.latest_result = None

if "last_error" not in st.session_state:
    st.session_state.last_error = None


# ============================================================
# 5. GENERATE SIMULATED SENSOR READING
# ============================================================

def generate_reading():
    """
    Generate one simulated machine sensor reading.
    """

    state = st.session_state.machine_state

    state["air"] = min(
        320.0,
        max(
            290.0,
            state["air"] + random.uniform(-0.7, 0.7),
        ),
    )

    state["speed"] = min(
        2200.0,
        max(
            900.0,
            state["speed"] + random.uniform(-60, 60),
        ),
    )

    state["torque"] = min(
        75.0,
        max(
            15.0,
            state["torque"] + random.uniform(-3, 3),
        ),
    )

    state["wear"] = min(
        240.0,
        state["wear"] + random.uniform(0, 0.5),
    )

    process_temp = state["air"] + random.uniform(8.0, 12.0)

    return {
        "product_type": "M",
        "air_temperature": round(state["air"], 2),
        "process_temperature": round(process_temp, 2),
        "rotational_speed": round(state["speed"], 2),
        "torque": round(state["torque"], 2),
        "tool_wear": round(state["wear"], 2),
    }


# ============================================================
# 6. API FUNCTIONS
# ============================================================

def fetch_records(limit=100):
    response = requests.get(
        f"{API_URL}/history",
        params={"limit": limit},
        timeout=5,
    )

    response.raise_for_status()

    return response.json().get("records", [])


def fetch_alerts(limit=100):
    response = requests.get(
        f"{API_URL}/alerts",
        params={"limit": limit},
        timeout=5,
    )

    response.raise_for_status()

    return response.json().get("alerts", [])


def fetch_statistics():
    response = requests.get(
        f"{API_URL}/statistics",
        timeout=5,
    )

    response.raise_for_status()

    return response.json()


def fetch_health():
    response = requests.get(
        f"{API_URL}/health",
        timeout=5,
    )

    response.raise_for_status()

    return response.json()


def submit_prediction(reading):
    response = requests.post(
        f"{API_URL}/predict",
        json=reading,
        timeout=5,
    )

    response.raise_for_status()

    return response.json()


# ============================================================
# 7. START / STOP CONTROLS
# ============================================================

control_col, status_col = st.columns([1, 2])

with control_col:

    if st.button(
        "▶ Start Monitoring",
        type="primary",
        use_container_width=True,
    ):
        st.session_state.monitoring = True
        st.session_state.last_error = None

    if st.button(
        "⏹ Stop Monitoring",
        use_container_width=True,
    ):
        st.session_state.monitoring = False


with status_col:

    if st.session_state.monitoring:
        st.success("🟢 Monitoring is running")
    else:
        st.info("⚪ Monitoring is stopped")


# ============================================================
# 8. LIVE DASHBOARD
# ============================================================

@st.fragment(run_every="3s")
def render_dashboard():

    # --------------------------------------------------------
    # 8A. Generate new prediction
    # --------------------------------------------------------

    if st.session_state.monitoring:

        try:

            reading = generate_reading()

            result = submit_prediction(reading)

            st.session_state.latest_reading = reading
            st.session_state.latest_result = result
            st.session_state.last_error = None

        except requests.RequestException as exc:

            st.session_state.last_error = str(exc)


    # --------------------------------------------------------
    # 8B. API error
    # --------------------------------------------------------

    if st.session_state.last_error:

        st.error(
            "❌ Could not communicate with the prediction API.\n\n"
            f"Ensure FastAPI is running at {API_URL}.\n\n"
            f"Details: {st.session_state.last_error}"
        )


    # --------------------------------------------------------
    # 8C. Load persistent backend information
    # --------------------------------------------------------

    try:

        stats = fetch_statistics()

        records = fetch_records(limit=100)

        alerts = fetch_alerts(limit=100)

        health = fetch_health()

    except requests.RequestException as exc:

        st.error(
            "Unable to load saved predictions.\n\n"
            "Please check that the FastAPI backend is running.\n\n"
            f"Details: {exc}"
        )

        return


    history = pd.DataFrame(records)

    alerts_df = pd.DataFrame(alerts)


    # ========================================================
    # 9. SYSTEM STATUS
    # ========================================================

    st.subheader("🔌 System Status")

    s1, s2, s3, s4 = st.columns(4)

    if health.get("status") == "healthy":
        s1.success("API: Healthy")
    else:
        s1.warning("API: Degraded")

    if health.get("model_loaded"):
        s2.success("ML Model: Loaded")
    else:
        s2.error("ML Model: Not Loaded")

    if health.get("database"):
        s3.success("Database: Connected")
    else:
        s3.error("Database: Disconnected")

    threshold = stats.get(
        "failure_threshold",
        health.get("failure_threshold", 0.40),
    )

    s4.info(
        f"Failure threshold: {threshold * 100:.0f}%"
    )


    # ========================================================
    # 10. PREDICTION SUMMARY
    # ========================================================

    st.subheader("📊 Prediction Summary")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "All-time predictions",
        stats["total_predictions"],
    )

    c2.metric(
        "Normal predictions",
        stats["normal_predictions"],
    )

    c3.metric(
        "Predicted failures",
        stats["failure_predictions"],
    )

    c4.metric(
        "Average estimated risk",
        f"{stats['average_estimated_risk_percentage']:.2f}%",
    )

    st.caption(
        "Statistics cover all predictions stored in SQLite."
    )


    # ========================================================
    # 11. CURRENT MACHINE READINGS
    # ========================================================

    st.subheader("🌡️ Current Machine Readings")

    reading = st.session_state.latest_reading

    result = st.session_state.latest_result


    if reading is not None and result is not None:

        a, b, c, d = st.columns(4)

        a.metric(
            "Air Temperature",
            f"{reading['air_temperature']:.2f} K",
        )

        b.metric(
            "Process Temperature",
            f"{reading['process_temperature']:.2f} K",
        )

        c.metric(
            "Rotational Speed",
            f"{reading['rotational_speed']:.0f} RPM",
        )

        d.metric(
            "Torque",
            f"{reading['torque']:.2f} Nm",
        )


        a, b, c, d = st.columns(4)

        a.metric(
            "Tool Wear",
            f"{reading['tool_wear']:.2f} min",
        )

        risk = result["estimated_failure_percentage"]

        b.metric(
            "Estimated Failure Risk",
            f"{risk:.2f}%",
        )

        c.metric(
            "Prediction Threshold",
            f"{result.get('failure_threshold', threshold) * 100:.0f}%",
        )

        d.metric(
            "Prediction ID",
            result.get("id", "N/A"),
        )


        # ----------------------------------------------------
        # Risk interpretation
        # ----------------------------------------------------

        if result["predicted_failure"] == 1:

            st.error(
                f"🚨 FAILURE RISK DETECTED — Estimated risk: "
                f"{risk:.2f}%"
            )

            st.warning(
                "The machine-learning model flagged this reading "
                "as a possible failure. This is an experimental "
                "prediction and not a confirmed machine fault."
            )

        elif risk >= threshold * 100 * 0.75:

            st.warning(
                f"⚠️ ELEVATED RISK — Estimated risk: "
                f"{risk:.2f}%"
            )

        else:

            st.success(
                f"🟢 NORMAL — Estimated risk: "
                f"{risk:.2f}%"
            )


    else:

        st.info(
            "Click Start Monitoring to generate simulated readings. "
            "Previously saved records are shown below."
        )


    # ========================================================
    # 12. SENSOR TRENDS
    # ========================================================

    st.subheader("📈 Sensor Trends")

    if not history.empty:

        chart_data = history.sort_values("id").copy()

        chart_data = chart_data.rename(
            columns={
                "air_temperature": "Air Temperature (K)",
                "process_temperature": "Process Temperature (K)",
                "rotational_speed": "Speed (RPM)",
                "tool_wear": "Tool Wear (min)",
                "torque": "Torque (Nm)",
                "failure_probability": "Failure Risk",
            }
        )

        chart_data = chart_data.set_index("id")


        st.markdown("**🌡️ Temperature trends**")

        temperature_columns = [
            "Air Temperature (K)",
            "Process Temperature (K)",
        ]

        available_temperature = [
            col
            for col in temperature_columns
            if col in chart_data.columns
        ]

        if available_temperature:

            st.line_chart(
                chart_data[available_temperature]
            )


        st.markdown(
            "**⚙️ Speed, torque and tool-wear trends**"
        )

        machine_columns = [
            "Speed (RPM)",
            "Torque (Nm)",
            "Tool Wear (min)",
        ]

        available_machine = [
            col
            for col in machine_columns
            if col in chart_data.columns
        ]

        if available_machine:

            st.line_chart(
                chart_data[available_machine]
            )


        st.markdown("**🚨 Failure-risk trend**")

        if "Failure Risk" in chart_data.columns:

            risk_chart = chart_data[
                ["Failure Risk"]
            ].copy()

            risk_chart["Failure Risk"] = (
                risk_chart["Failure Risk"] * 100
            )

            st.line_chart(risk_chart)


    else:

        st.info(
            "No prediction history is available yet."
        )


    # ========================================================
    # 13. RECENT PREDICTION HISTORY
    # ========================================================

    st.subheader("🗂️ Recent Prediction History")

    if not history.empty:

        display_history = history.copy()

        display_history["Failure Risk (%)"] = (
            display_history["failure_probability"] * 100
        ).round(2)

        display_history["Prediction"] = (
            display_history["predicted_failure"]
            .map({
                0: "Normal",
                1: "Failure",
            })
        )

        display_columns = [
            "id",
            "timestamp",
            "product_type",
            "air_temperature",
            "process_temperature",
            "rotational_speed",
            "torque",
            "tool_wear",
            "Failure Risk (%)",
            "Prediction",
        ]

        display_history = display_history[
            [
                col
                for col in display_columns
                if col in display_history.columns
            ]
        ]

        st.dataframe(
            display_history,
            use_container_width=True,
            hide_index=True,
        )


        # CSV export

        csv_data = history.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="⬇️ Download Loaded History as CSV",
            data=csv_data,
            file_name="predictive_maintenance_history.csv",
            mime="text/csv",
        )

        st.caption(
            "The table contains the latest 100 records. "
            "All-time statistics are calculated separately."
        )

    else:

        st.info(
            "No prediction records have been saved yet."
        )


    # ========================================================
    # 14. FAILURE ALERT HISTORY
    # ========================================================

    st.subheader("🚨 Failure Alert History")

    if not alerts_df.empty:

        alerts_display = alerts_df.copy()

        if "failure_probability" in alerts_display.columns:

            alerts_display["Failure Risk (%)"] = (
                alerts_display["failure_probability"] * 100
            ).round(2)

        if "predicted_failure" in alerts_display.columns:

            alerts_display["Prediction"] = (
                alerts_display["predicted_failure"]
                .map({
                    0: "Normal",
                    1: "Failure",
                })
            )

        st.dataframe(
            alerts_display,
            use_container_width=True,
            hide_index=True,
        )


        alerts_csv = alerts_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="⬇️ Download Failure Alerts as CSV",
            data=alerts_csv,
            file_name="predictive_maintenance_alerts.csv",
            mime="text/csv",
            key="download_alerts",
        )

    else:

        st.info(
            "No failure alerts have been saved in the database."
        )


    # ========================================================
    # 15. RISK INTERPRETATION
    # ========================================================

    st.subheader("ℹ️ Risk Interpretation")

    st.write(
        f"""
        **Current classification threshold:** {threshold * 100:.0f}%

        The ML model produces an estimated probability of machine
        failure. A prediction is classified as a failure when the
        probability reaches the configured threshold.

        The probability is an experimental model output and is not
        a calibrated guarantee that a machine will fail.
        """
    )

    st.caption(
        "This project uses simulated sensor readings and the "
        "AI4I 2020 dataset. It is a software prototype and has "
        "not been validated for real industrial safety or "
        "maintenance decisions."
    )


# ============================================================
# 16. RUN DASHBOARD
# ============================================================

render_dashboard()