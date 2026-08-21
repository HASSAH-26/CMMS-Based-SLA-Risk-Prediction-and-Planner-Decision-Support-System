import streamlit as st
import joblib
import os
import pandas as pd
from datetime import date


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CMMS SLA Risk Prediction",
    page_icon="🛠️",
    layout="wide"
)


# ============================================================
# FILE NAMES
# ============================================================

MODEL_FILE = "cmms_sla_risk_model.pkl"
COLUMNS_FILE = "cmms_model_columns.pkl"


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = joblib.load(
        MODEL_FILE
    )

    model_columns = joblib.load(
        COLUMNS_FILE
    )

    return model, model_columns


# ============================================================
# CHECK FILES
# ============================================================

if not os.path.exists(MODEL_FILE):

    st.error(
        f"❌ Cannot find {MODEL_FILE}"
    )

    st.stop()


if not os.path.exists(COLUMNS_FILE):

    st.error(
        f"❌ Cannot find {COLUMNS_FILE}"
    )

    st.stop()


# ============================================================
# LOAD MODEL
# ============================================================

try:

    model, model_columns = load_model()

except Exception as e:

    st.error(
        "❌ Model loading failed."
    )

    st.exception(e)

    st.stop()


# ============================================================
# HEADER
# ============================================================

st.title(
    "🛠️ CMMS SLA Risk Prediction"
)

st.subheader(
    "Planner Decision Support System"
)

st.write(
    """
    Enter the information available when a maintenance
    work request is created. The machine learning model
    will estimate the probability of SLA breach.
    """
)

st.divider()


# ============================================================
# MODEL STATUS
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:

    st.success(
        "✅ Model Loaded"
    )

with col2:

    st.info(
        f"Model: {type(model.named_steps['classifier']).__name__}"
    )

with col3:

    st.info(
        f"Model Features: {len(model_columns)}"
    )


# ============================================================
# PLANNER FORM
# ============================================================

st.header(
    "📋 New Maintenance Work Request"
)


with st.form("sla_prediction_form"):

    # ========================================================
    # 1. BASIC INFORMATION
    # ========================================================

    st.subheader(
        "1️⃣ Basic Information"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        building = st.selectbox(
            "Building",
            [
                "Building A",
                "Building B",
                "Building C",
                "Building D",
                "Building E",
                "Building F",
                "Building G",
                "Building H"
            ]
        )

    with col2:

        department = st.selectbox(
            "Department",
            [
                "Administration",
                "Data Center",
                "Emergency",
                "ICU",
                "Kitchen",
                "Laboratory",
                "Laundry",
                "Office",
                "Pharmacy"
            ]
        )

    with col3:

        asset_category = st.selectbox(
            "Asset Category",
            [
                "Civil",
                "ELV",
                "Electrical",
                "Fire",
                "HVAC",
                "Plumbing"
            ]
        )


    col1, col2, col3 = st.columns(3)

    with col1:

        asset_type = st.selectbox(
            "Asset Type",
            [
                "AHU",
                "Access Control",
                "CCTV",
                "Chiller",
                "Door",
                "FCU",
                "Fire Alarm",
                "Furniture",
                "Lighting",
                "Piping/Fixtures",
                "Power Supply",
                "Pump",
                "Sprinkler",
                "UPS",
                "Wall/Paint"
            ]
        )

    with col2:

        complaint_type = st.selectbox(
            "Complaint Type",
            [
                "Abnormal Noise",
                "Airflow Weak",
                "Automatic Door Sensor Fault",
                "Ballast Failure",
                "Battery Fault",
                "Beeping Sound",
                "Belt Broken",
                "Biometric Sensor Error",
                "Blower Fan Fault",
                "Blurred Image",
                "Break Glass Triggered",
                "Breaker Trip",
                "Bypass Active Warning",
                "Cabinet Lock Stuck",
                "Camera Offline",
                "Card Reader Fault",
                "Chair Broken",
                "Check Valve Failure",
                "Chilled Water Low Flow",
                "Compressor Trip",
                "Condenser Fan Defective",
                "Control Valve Tamper",
                "DB Panel Overheat",
                "Damper Actuator Stuck",
                "Desk Damaged",
                "Detector Fault",
                "Dimmer Fault",
                "Door Closer Leaking",
                "Door Damaged",
                "Door Magnet Fault",
                "Door Not Opening",
                "Drain Clogged",
                "Drawer Runner Fault",
                "Duct Noise",
                "Earth Leakage Trip",
                "Emergency Light Down",
                "False Alarm",
                "False Ceiling Sagging",
                "Faucet Leak",
                "Filter Clogged",
                "Flickering",
                "Flow Switch Fault",
                "Flush Valve Fault",
                "Gate Barrier Stuck",
                "Grout Missing",
                "Handle Loose",
                "High Pressure",
                "High Temp",
                "Hinge Broken",
                "IR Night Vision Fault",
                "Inverter Failure",
                "Jockey Pump Continuous Run",
                "Light Not Working",
                "Lock Fault",
                "Low Pressure",
                "MCP Glass Broken",
                "Motor Overheat",
                "No Cooling",
                "No Recording",
                "Output Voltage Low",
                "Overload Trip",
                "PTZ Control Fault",
                "Paint Peeling",
                "Panel Fault",
                "Partition Loose",
                "Pipe Burst",
                "Pipe Leakage",
                "Power Outage",
                "Pressure Drop",
                "Pump Trip",
                "Refrigerant Leak",
                "Seal Failure",
                "Sensor Fault",
                "Short Circuit",
                "Socket Burned",
                "Strobe Light Defective",
                "Thermostat Fault",
                "Tile Broken",
                "Toilet Overflow",
                "UPS Alarm",
                "Vibration High",
                "Video Loss Signal",
                "Voltage Fluctuation",
                "Wall Crack",
                "Water Heater Fault",
                "Water Leakage",
                "Water Seepage",
                "Zone Isolation"
            ]
        )

    with col3:

        maintenance_type = st.selectbox(
            "Maintenance Type",
            [
                "Corrective",
                "Preventive"
            ]
        )


    # ========================================================
    # 2. PRIORITY
    # ========================================================

    st.subheader(
        "2️⃣ Priority & Criticality"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        priority = st.selectbox(
            "Priority",
            [
                "P1",
                "P2",
                "P3",
                "P4",
                "P5"
            ]
        )

        st.caption(
            "P1 = Highest priority | P5 = Lowest priority"
        )

    with col2:

        criticality = st.selectbox(
            "Criticality",
            [
                "Critical",
                "High",
                "Medium",
                "Low"
            ]
        )

    with col3:

        emergency = st.selectbox(
            "Emergency",
            [
                "No",
                "Yes"
            ]
        )


    # ========================================================
    # 3. EQUIPMENT
    # ========================================================

    st.subheader(
        "3️⃣ Equipment Information"
    )

    col1, col2 = st.columns(2)

    with col1:

        equipment_age = st.number_input(
            "Equipment Age (Years)",
            min_value=0,
            max_value=100,
            value=10,
            step=1
        )

    with col2:

        equipment_age_group = st.selectbox(
            "Equipment Age Group",
            [
                "0-5",
                "6-10",
                "11-15",
                "16-20",
                "21+"
            ]
        )


    # ========================================================
    # 4. TECHNICIAN & RESOURCES
    # ========================================================

    st.subheader(
        "4️⃣ Technician & Resources"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        technician_trade = st.selectbox(
            "Technician Trade",
            [
                "Civil",
                "ELV",
                "Electrical",
                "Fire",
                "HVAC",
                "Plumbing"
            ]
        )

    with col2:

        technician_count = st.number_input(
            "Technician Count",
            min_value=1,
            max_value=100,
            value=1,
            step=1
        )

    with col3:

        spare_required = st.selectbox(
            "Spare Required",
            [
                "No",
                "Yes"
            ]
        )


    col1, col2 = st.columns(2)

    with col1:

        ptw_required = st.selectbox(
            "PTW Required",
            [
                "No",
                "Yes"
            ]
        )

    with col2:

        estimated_cost = st.number_input(
            "Estimated Cost",
            min_value=0.0,
            value=500.0,
            step=100.0
        )


    # ========================================================
    # 5. SLA & ENVIRONMENT
    # ========================================================

    st.subheader(
        "5️⃣ SLA & Environment"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        response_target = st.number_input(
            "Response Target (Minutes)",
            min_value=1,
            value=60,
            step=1
        )

    with col2:

        outside_temperature = st.number_input(
            "Outside Temperature (°C)",
            min_value=-20.0,
            max_value=70.0,
            value=30.0,
            step=0.5
        )

    with col3:

        technician_available = st.selectbox(
            "Technician Available",
            [
                "Yes",
                "No"
            ]
        )


    # ========================================================
    # 6. REQUEST DATE & TIME
    # ========================================================

    st.subheader(
        "6️⃣ Request Date & Time"
    )

    col1, col2 = st.columns(2)

    with col1:

        request_date = st.date_input(
            "Request Date",
            value=date.today()
        )

    with col2:

        request_hour = st.number_input(
            "Request Hour",
            min_value=0,
            max_value=23,
            value=10,
            step=1
        )


    # ========================================================
    # PREDICT BUTTON
    # ========================================================

    st.divider()

    submitted = st.form_submit_button(
        "🔮 Predict SLA Risk",
        use_container_width=True
    )


# ============================================================
# CREATE MODEL INPUT
# ============================================================

if submitted:

    # ========================================================
    # DATE FEATURES
    # ========================================================

    request_year = request_date.year

    request_month = request_date.month

    request_day = request_date.day

    request_dayofweek = request_date.weekday()

    request_weekend = (
        request_dayofweek >= 5
    )


    # ========================================================
    # CONVERT YES / NO
    # ========================================================

    emergency_value = (
        1 if emergency == "Yes"
        else 0
    )

    spare_value = (
        1 if spare_required == "Yes"
        else 0
    )

    ptw_value = (
        1 if ptw_required == "Yes"
        else 0
    )

    technician_available_value = (
        1 if technician_available == "Yes"
        else 0
    )


    # ========================================================
    # CREATE INPUT DATA
    # ========================================================

    input_data = {

        "Year": request_year,

        "Month": request_month,

        "Building": building,

        "Department": department,

        "Asset_Category": asset_category,

        "Asset_Type": asset_type,

        "Equipment_Age": equipment_age,

        "Criticality": criticality,

        "Complaint_Type": complaint_type,

        "Maintenance_Type": maintenance_type,

        "Priority": priority,

        "Emergency": emergency_value,

        "Technician_Trade": technician_trade,

        "Technician_Count": technician_count,

        "Spare_Required": spare_value,

        "PTW_Required": ptw_value,

        "Outside_Temperature": outside_temperature,

        "Response_Target_Min": response_target,

        "Estimated_Cost": estimated_cost,

        "Request_Year": request_year,

        "Request_Month": request_month,

        "Request_Day": request_day,

        "Request_DayOfWeek": request_dayofweek,

        "Request_Weekend": request_weekend,

        "Request_Hour": request_hour,

        "Equipment_Age_Group": equipment_age_group
    }


    # ========================================================
    # DATAFRAME
    # ========================================================

    input_df = pd.DataFrame(
        [input_data]
    )


    # ========================================================
    # CHECK MODEL COLUMNS
    # ========================================================

    missing_columns = [
        col
        for col in model_columns
        if col not in input_df.columns
    ]


    if missing_columns:

        st.error(
            "❌ Some model features are missing."
        )

        for col in missing_columns:

            st.write(
                f"- `{col}`"
            )

        st.stop()


    # ========================================================
    # SELECT EXACT COLUMNS
    # ========================================================

    model_input = input_df[
        model_columns
    ].copy()


    # ========================================================
    # PREDICTION
    # ========================================================

    try:

        prediction = model.predict(
            model_input
        )[0]


        probabilities = model.predict_proba(
            model_input
        )[0]


        classes = list(
            model.classes_
        )


        if 1 in classes:

            breach_probability = probabilities[
                classes.index(1)
            ]

        else:

            breach_probability = None


        # ====================================================
        # RESULT
        # ====================================================

        st.divider()

        st.header(
            "🎯 Prediction Result"
        )


        if prediction == 1:

            prediction_text = "SLA BREACH"

        else:

            prediction_text = "SLA MET"


        col1, col2 = st.columns(2)


        with col1:

            st.metric(
                "Prediction",
                prediction_text
            )


        with col2:

            if breach_probability is not None:

                st.metric(
                    "SLA Breach Probability",
                    f"{breach_probability * 100:.2f}%"
                )

            else:

                st.metric(
                    "SLA Breach Probability",
                    "Not available"
                )


        # ====================================================
        # RISK LEVEL
        # ====================================================

        if breach_probability is not None:

            if breach_probability >= 0.70:

                risk_level = "HIGH"

                st.error(
                    "🔴 HIGH SLA BREACH RISK"
                )


            elif breach_probability >= 0.40:

                risk_level = "MEDIUM"

                st.warning(
                    "🟠 MEDIUM SLA BREACH RISK"
                )


            else:

                risk_level = "LOW"

                st.success(
                    "🟢 LOW SLA BREACH RISK"
                )


            # =================================================
            # RECOMMENDATION
            # =================================================

            st.subheader(
                "Planner Recommendation"
            )


            if risk_level == "HIGH":

                st.write(
                    """
                    ⚠️ **Immediate attention recommended.**

                    - Check technician availability
                    - Check spare/material availability
                    - Consider escalation
                    - Monitor response closely
                    - Take preventive action before SLA breach
                    """
                )


            elif risk_level == "MEDIUM":

                st.write(
                    """
                    ⚠️ **Close monitoring recommended.**

                    - Check resource availability
                    - Monitor response time
                    - Verify material availability
                    - Follow up with the assigned team
                    """
                )


            else:

                st.write(
                    """
                    ✅ **Normal monitoring recommended.**

                    The current request has a relatively
                    lower predicted probability of SLA breach.
                    """
                )


        # ====================================================
        # MODEL INPUT
        # ====================================================

        with st.expander(
            "🔎 View Exact Data Sent to Model"
        ):

            st.dataframe(
                model_input,
                use_container_width=True
            )


        # ====================================================
        # DISCLAIMER
        # ====================================================

        st.caption(
            """
            Note: This prediction is a machine-learning-based
            decision-support estimate and should be considered
            together with operational judgement and actual CMMS
            information.
            """
        )


    except Exception as e:

        st.error(
            "❌ Prediction failed."
        )

        st.exception(e)

        st.write(
            "### Data sent to the model"
        )

        st.dataframe(
            model_input,
            use_container_width=True
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "CMMS-Based SLA Risk Prediction | "
    "Planner Decision Support System"
)