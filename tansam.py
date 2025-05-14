import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import requests
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
st.set_page_config(layout="wide")

# ---------- CONFIGURATION ----------
API_KEY = "pp9o7Z8QzIpy05pZj5dg6cYwyx5yMzLs"  # Tomorrow.io API Key
LAT, LON = 13.0827, 80.2707  # Tamil Nadu
file_path = "PROJECT_DATASET.xlsx"

# ---------- FUNCTIONS ----------
@st.cache_data
def load_data(path):
   df = pd.read_excel("PROJECT_DATASET.xlsx", engine='openpyxl')
   return df

def get_weather_data(lat, lon, api_key):
    url = f"https://api.tomorrow.io/v4/weather/forecast?location={lat},{lon}&apikey={api_key}"
    res = requests.get(url)
    data = res.json()
    try:
        interval = data["timelines"]["hourly"][0]["values"]
        return interval["temperature"], interval["humidity"], interval["windSpeed"]
    except KeyError:
        raise ValueError("Unexpected API response structure")

# ---------- LOAD AND CLEAN ----------
data = load_data(file_path)
data.rename(columns={
    'new_residential_units': 'NewResidentialUnits',
    'commercial_space_added_sqm': 'CommercialSpaceAdded',
    'population_growth': 'PopulationGrowth',
    'real_estate_index': 'RealEstateIndex',
    'date': 'Date',
    'time': 'Time',
    'datetime_iso': 'DatetimeISO',
    'wind_generation_mw': 'WindGeneration',
    'installed_capacity_mw': 'InstalledCapacity',
    'is_holiday': 'Holiday',
    'temperature_c': 'Temperature',
    'humidity_percent': 'Humidity',
    'radiation_wm2': 'Radiation',
    'wind_speed_kmph': 'WindSpeed',
    'rainfall_mm': 'Rainfall',
    'power_purchased_mw': 'PowerPurchased',
    'source_type': 'SourceType',
    'cost_per_mw': 'CostPerMW',
    'industrial_load_mw': 'IndustrialLoad',
    'commercial_load_mw': 'CommercialLoad',
    'residential_load_mw': 'ResidentialLoad',
    'solar_generation_mw': 'SolarGeneration',
    'solar_irradiance': 'SolarIrradiance',
    'reserve_generation_mw': 'ReserveGeneration',
    'grid_balancing_mw': 'GridBalancing',
    'total_demand_mw': 'Demand',
    'region': 'Region',
    'peak_demand_flag': 'PeakDemandFlag'
}, inplace=True)

data['Datetime'] = pd.to_datetime(data['DatetimeISO'])
data.fillna(data.mean(numeric_only=True), inplace=True)
data['Holiday'] = data['Holiday'].apply(lambda x: 1 if str(x).strip().lower() in ['yes', '1', 'true'] else 0)

# ---------- MODEL ----------
features = ['Temperature', 'Humidity', 'WindSpeed', 'Holiday', 'RealEstateIndex']
X = data[features]
y = data['Demand']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = LinearRegression().fit(X_train_scaled, y_train)
mse = mean_squared_error(y_test, model.predict(X_test_scaled))

# ---------- PREDICTION ----------
data['Predicted'] = model.predict(scaler.transform(data[features]))

# ---------- STREAMLIT UI ----------

st.title("🔌 Electricity Demand Forecast Dashboard")
st.markdown(f"**Model Mean Squared Error:** `{mse:.2f}`")

# ---------- LINE GRAPH: Actual vs Predicted ----------
fig1 = px.line(data, x='Datetime', y=['Demand', 'Predicted'],
               labels={'value': 'MW', 'variable': 'Legend'},
               title="Actual vs Predicted Demand Over Time")
fig1.update_traces(mode="lines")
fig1.update_layout(hovermode="x unified")
fig1.update_xaxes(tickformat="%d-%b %I%p")
st.plotly_chart(fig1, use_container_width=True)

# ---------- DUCK CURVE ----------
st.subheader("🦆 Duck Curve Analysis")
data['Hour'] = data['Datetime'].dt.hour
avg_demand = data.groupby('Hour')['Demand'].mean().reset_index()

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=avg_demand['Hour'], y=avg_demand['Demand'],
                          mode='lines+markers', name='Avg Demand',
                          line=dict(color='orange', width=3)))
fig2.update_layout(title="Duck Curve - Avg Hourly Demand",
                   xaxis_title="Hour of Day",
                   yaxis_title="Average Demand (MW)")
st.plotly_chart(fig2, use_container_width=True)

# ---------- WEATHER + OPTIMIZATION SECTIONS ----------
st.subheader("🌤 Real-Time Weather & Optimization")

col1, col2 = st.columns(2)
with col1:
    try:
        temp, humidity, wind_speed = get_weather_data(LAT, LON, API_KEY)
        st.metric("Temperature (°C)", f"{temp:.2f}")
        st.metric("Humidity (%)", f"{humidity:.2f}")
        st.metric("Wind Speed (km/h)", f"{wind_speed:.2f}")
    except Exception as e:
        st.error("Weather fetch failed.")
        st.exception(e)

with col2:
    is_holiday = st.checkbox("Is today a holiday?")
    real_estate_index = st.slider("Real Estate Index", 50, 150, 100)
    user_input = pd.DataFrame([[temp, humidity, wind_speed, int(is_holiday), real_estate_index]],
                              columns=features)
    input_scaled = scaler.transform(user_input)
    predicted_now = model.predict(input_scaled)[0]
    st.metric("Predicted Demand Now (MW)", f"{predicted_now:.2f}")

# ---------- OPTIMIZATION SUGGESTION ----------
st.subheader("⚡ Source Optimization Recommendation")
if temp > 30 and humidity < 50:
    source = "Solar"
else:
    source = "Grid"
st.success(f"🔋 **Recommended Source:** `{source}` based on current weather")
