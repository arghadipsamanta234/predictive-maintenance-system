
import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
)

st.title("⚙️ Predictive Maintenance System")
st.caption("Machine monitoring and experimental failure prediction")

# Check backend connection
try:
    health_response = requests.get(
        f"{API_URL}/health", timeout=3
    )
    health_response.raise_for_status()
    st.success("Backend connected")
except requests.RequestException:
    st.error(
        "Backend is not reachable. Start the API in another terminal "
        "using: python -m uvicorn backend.main:app --reload"
    )
    st.stop()

st.divider()

st.subheader("Machine Operating Parameters")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        product_type = st.selectbox(
            "Product type", ["L", "M", "H"], index=1
        )
        air_temperature = st.number_input(
            "Air temperature (K)",
            min_value=250.0,
            max_value=350.0,
            value=298.1,
            step=0.1,
        )
        process_temperature = st.number_input(
            "Process temperature (K)",
            min_value=250.0,
            max_value=400.0,
            value=308.6,
            step=0.1,
        )

    with col2:
        rotational_speed = st.number_input(
            "Rotational speed (RPM)",
            min_value=1,
            max_value=10000,
            value=1500,
            step=10,
        )
        torque = st.number_input(
            "Torque (Nm)",
            min_value=0.0,
            max_value=1000.0,
            value=40.0,
            step=1.0,
        )
        tool_wear = st.number_input(
            "Tool wear (minutes)",
            min_value=0.0,
            max_value=1000.0,
            value=100.0,
            step=1.0,
        )

    submitted = st.form_submit_button(
        "Predict Machine Failure",
        type="primary",
        use_container_width=True,
    )

if submitted:
    payload = {
        "product_type": product_type,
        "air_temperature": air_temperature,
        "process_temperature": process_temperature,
        "rotational_speed": rotational_speed,
        "torque": torque,
        "tool_wear": tool_wear,
    }

    try:
        with st.spinner("Analyzing machine readings..."):
            response = requests.post(
                f"{API_URL}/predict",
                json=payload,
                timeout=15,
            )
            response.raise_for_status()
            result = response.json()

        st.divider()
        st.subheader("Prediction Results")

        metric1, metric2 = st.columns(2)

        with metric1:
            st.metric(
                "Estimated failure probability",
                f"{result['estimated_failure_percentage']:.2f}%",
            )

        with metric2:
            st.metric(
                "Predicted class",
                "Failure" if result["predicted_failure"] == 1
                else "No failure",
            )

        if result["predicted_failure"] == 1:
            st.warning(
                "The model predicts a failure. "
                "Review the readings and arrange an appropriate inspection."
            )
        else:
            st.info(
                "The model predicts no failure for these inputs. "
                "This does not guarantee safe operation."
            )

        st.caption(result["warning"])

        with st.expander("View submitted readings and API response"):
            st.json({
                "input": payload,
                "prediction": result,
            })

    except requests.RequestException as error:
        st.error(f"Could not get a prediction from the API: {error}")
    except (KeyError, ValueError) as error:
        st.error(f"Unexpected API response: {error}")

st.divider()
st.caption(
    "Prototype using the synthetic AI4I 2020 dataset. "
    "Not validated for real industrial safety decisions."
)
