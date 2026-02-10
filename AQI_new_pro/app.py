import streamlit as st
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

st.set_page_config(page_title="AQI Forecast App", layout="wide")
st.title("🌫️ Air Quality Index (AQI) Forecast System")

# ===================== AQI Category Function =====================
def aqi_category(aqi):
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Satisfactory"
    elif aqi <= 200:
        return "Moderate"
    elif aqi <= 300:
        return "Poor"
    elif aqi <= 400:
        return "Very Poor"
    else:
        return "Severe"

# ===================== Sidebar Navigation =====================
page = st.sidebar.selectbox("Select Page", ["City Forecast", "City Comparison (Power BI)"])

# ===================== CITY FORECAST PAGE =====================
if page == "City Forecast":

    city = st.sidebar.selectbox("Choose a city", ["Delhi", "Bengaluru", "TVM"])

    def load_city(city_name):
        if city_name == "Delhi":
            model = load_model("lstm_delhi.h5", compile=False)
            with open("scaler_delhi.pkl", "rb") as f:
                scaler = pickle.load(f)
            df = pd.read_csv("delhi_aqi.csv")

        elif city_name == "Bengaluru":
            model = load_model("lstm_blr.h5", compile=False)
            with open("scaler_blr.pkl", "rb") as f:
                scaler = pickle.load(f)
            df = pd.read_csv("blr_aqi.csv")

        else:  # TVM
            model = load_model("lstm_tvm.h5", compile=False)
            with open("scaler_tvm.pkl", "rb") as f:
                scaler = pickle.load(f)
            df = pd.read_csv("tvm_aqi.csv")

        # If CSV got merged into one column, split it
        if len(df.columns) == 1:
            col = df.columns[0]
            if df[col].astype(str).str.contains(";").any():
                df = df[col].astype(str).str.split(";", expand=True)
            else:
                df = df[col].astype(str).str.split(",", expand=True)
            df.columns = df.iloc[0]
            df = df.drop(0).reset_index(drop=True)

        # ---- Use the real 'date' column (MM/DD/YYYY) ----
        if "date" not in df.columns:
            st.error("❌ 'date' column not found in CSV.")
            st.write("Columns found:", df.columns.tolist())
            st.stop()

        df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y", errors="coerce")
        df = df.dropna(subset=["date"])
        df = df.set_index("date")

        # ---- Ensure AQI column exists ----
        if "AQI" not in df.columns:
            for c in df.columns:
                if c.strip().lower() == "aqi":
                    df = df.rename(columns={c: "AQI"})
                    break

        if "AQI" not in df.columns:
            st.error("❌ Could not find 'AQI' column in the CSV.")
            st.stop()

        # Ensure AQI numeric
        df["AQI"] = pd.to_numeric(df["AQI"], errors="coerce")
        df = df.dropna(subset=["AQI"])

        return model, scaler, df.sort_index()

    # Load selected city
    model, scaler, data = load_city(city)
    st.subheader(f"📍 Selected City: {city}")

    # Prepare AQI series
    aqi_values = data[["AQI"]].values
    aqi_scaled = scaler.transform(aqi_values)

    # Take last 14 days
    last_14 = aqi_scaled[-14:]

    # Predict
    pred = model.predict(last_14.reshape(1, 14, 1), verbose=0)

    # If model outputs 7 days directly
    if len(pred.shape) == 2 and pred.shape[1] == 7:
        future_preds_scaled = pred.reshape(-1, 1)
    else:
        # Recursive forecasting (1-day ahead model)
        future_preds_scaled = []
        current_input = last_14.copy()
        for _ in range(7):
            p = model.predict(current_input.reshape(1, 14, 1), verbose=0)
            val = p[0, 0]
            future_preds_scaled.append(val)
            current_input = np.vstack([current_input[1:], [[val]]])
        future_preds_scaled = np.array(future_preds_scaled).reshape(-1, 1)

    # Inverse scale
    future_preds = scaler.inverse_transform(future_preds_scaled)

    # Create future dates
    last_date = pd.to_datetime(data.index.max())
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=7)

    # ===================== PLOTS =====================
    st.subheader("📈 Historical AQI Trend")
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(data.index[-100:], data["AQI"].tail(100), color="blue")
    ax1.set_title(f"Historical AQI - {city}")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("AQI")
    ax1.grid(True)
    st.pyplot(fig1)

    st.subheader("🔮 AQI Forecast (Next 7 Days)")
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(future_dates, future_preds, "o--", color="orange")
    ax2.set_title(f"7-Day AQI Forecast - {city}")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("AQI")
    ax2.grid(True)
    st.pyplot(fig2)

    st.subheader("📊 Historical + Forecast AQI (Combined View)")
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    ax3.plot(data.index[-100:], data["AQI"].tail(100), label="Historical AQI", color="blue")
    ax3.plot(future_dates, future_preds, "o--", label="Forecast AQI", color="orange")
    ax3.set_title(f"AQI Trend and Forecast - {city}")
    ax3.set_xlabel("Date")
    ax3.set_ylabel("AQI")
    ax3.legend()
    ax3.grid(True)
    st.pyplot(fig3)

    # ===================== TABLE =====================
    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Predicted AQI": future_preds.flatten()
    })
    forecast_df["Category"] = forecast_df["Predicted AQI"].apply(aqi_category)

    st.subheader("📊 7-Day AQI Forecast (Table)")
    st.dataframe(forecast_df, use_container_width=True)

    # ===================== DOWNLOAD BUTTON =====================
    csv_data = forecast_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Forecast as CSV",
        data=csv_data,
        file_name=f"{city.lower()}_7day_aqi_forecast.csv",
        mime="text/csv"
    )

# ===================== POWER BI PAGE =====================
if page == "City Comparison (Power BI)":
    st.title("📊 City-wise AQI Comparison Dashboard")

    st.write("""
    This page shows the **Power BI dashboard screenshot** for city-wise AQI comparison.

    The dashboard includes:
    - Delhi vs Bengaluru vs TVM AQI trends
    - Historical comparison
    - Seasonal patterns
    - Summary KPIs
    """)

    st.subheader("📌 Power BI Dashboard")
    st.image("aqi_powerbi.png", use_container_width=True)
