# karachi_energy_dashboard_updated_with_accuracy_fixed_v3.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from io import BytesIO
import warnings

warnings.filterwarnings("ignore")
st.set_page_config(page_title="K-Electric Dashboard", layout="wide")

# ----------------------------
# Helpers
# ----------------------------
@st.cache_data
def load_cleaned_data(path="Karachi_Energy_Cleaned.csv"):
    df = pd.read_csv(path, parse_dates=['Date'])
    df.rename(columns=lambda c: c.strip(), inplace=True)
    df['Device Description'] = df['Device Description'].astype(str)
    df['Grid'] = df['Grid'].astype(str)
    df = df.dropna(subset=['Date'])
    return df

def prepare_series_for_model(df_device):
    series = df_device.copy()
    if 'Date' in series.columns:
        series = series.set_index('Date')
    series = series.sort_index()
    series['Reading'] = pd.to_numeric(series['Reading'], errors='coerce')
    series = series.dropna(subset=['Reading'])
    return series['Reading']

def arima_forecast(series, periods, order=(1,1,1)):
    model = ARIMA(series, order=order)
    fitted = model.fit()
    forecast = fitted.forecast(steps=periods)
    last_date = series.index.max()
    future_index = pd.date_range(start=last_date, periods=periods+1, freq='D')[1:]
    forecast_df = pd.DataFrame({'forecast': forecast.values}, index=future_index)
    return forecast_df

def prepare_lstm_series(series, n_lags=7):
    values = series.values.reshape(-1,1)
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled = scaler.fit_transform(values)
    X, y = [], []
    for i in range(n_lags, len(scaled)):
        X.append(scaled[i-n_lags:i, 0])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y, scaler

def lstm_forecast(series, periods=7, n_lags=7, epochs=15, batch_size=32):
    X, y, scaler = prepare_lstm_series(series, n_lags)
    model = Sequential()
    model.add(LSTM(20, activation='relu', input_shape=(X.shape[1], X.shape[2])))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
    last_seq = X[-1]
    forecasts = []
    for _ in range(periods):
        pred = model.predict(last_seq.reshape(1, n_lags, 1), verbose=0)
        forecasts.append(pred[0,0])
        last_seq = np.roll(last_seq, -1)
        last_seq[-1] = pred
    forecasts = scaler.inverse_transform(np.array(forecasts).reshape(-1,1)).flatten()
    last_date = series.index.max()
    future_index = pd.date_range(start=last_date, periods=periods+1, freq='D')[1:]
    return pd.DataFrame({'forecast': forecasts}, index=future_index)

def calculate_accuracy(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1e-6, None))) * 100
    accuracy = 100 - mape
    return max(0, round(accuracy,2))

def to_downloadable_csv(df):
    buffer = BytesIO()
    df.to_csv(buffer, index=True)
    buffer.seek(0)
    return buffer

# ----------------------------
# Sidebar Controls
# ----------------------------
st.sidebar.header("Data & Forecast Settings")
data_file = st.sidebar.text_input("Clean CSV path", "Karachi_Energy_Cleaned.csv")
df_all = load_cleaned_data(data_file)

grids = ["All Grids"] + sorted(df_all['Grid'].unique())
selected_grid = st.sidebar.selectbox("Select Grid", options=grids)

if selected_grid != "All Grids":
    meters_in_grid = sorted(df_all[df_all['Grid']==selected_grid]['Device Description'].unique())
else:
    meters_in_grid = sorted(df_all['Device Description'].unique())

selected_meter = st.sidebar.selectbox("Select Meter", options=meters_in_grid)
horizon = st.sidebar.radio("Forecast horizon (days)", options=[7,15], index=0)
model_choice = st.sidebar.multiselect("Model(s) to use", options=['ARIMA','LSTM'], default=['ARIMA','LSTM'])
run_button = st.sidebar.button("Run Forecast")

# ----------------------------
# Updated Pie Chart Section
# ----------------------------
st.title(f"⚡ K-Electric Forecast Dashboard")

if selected_grid == "All Grids":
    st.subheader("Total Energy Consumption by Grids (All Grids)")
    grid_sum = df_all.groupby('Grid')['Reading'].sum().sort_values(ascending=False)
    fig_pie, ax_pie = plt.subplots(figsize=(6,6))
    ax_pie.pie(grid_sum, labels=grid_sum.index, autopct='%1.1f%%', startangle=140, shadow=True)
    ax_pie.set_title("Total Consumption by Grids", fontsize=14)
    st.pyplot(fig_pie)
    st.metric(label="Total Energy Consumption (All Grids)", value=f"{grid_sum.sum():,.0f} kWh")
else:
    st.subheader(f"Total Energy Consumption by Meters in {selected_grid}")
    grid_data = df_all[df_all['Grid'] == selected_grid]
    meter_sum = grid_data.groupby('Device Description')['Reading'].sum().sort_values(ascending=False)
    fig_pie, ax_pie = plt.subplots(figsize=(6,6))
    ax_pie.pie(meter_sum, labels=meter_sum.index, autopct='%1.1f%%', startangle=140, shadow=True)
    ax_pie.set_title(f"Total Consumption by Meters ({selected_grid})", fontsize=14)
    st.pyplot(fig_pie)
    total_grid_consumption = meter_sum.sum()
    st.metric(label=f"Total Energy Consumption of {selected_grid}", value=f"{total_grid_consumption:,.0f} kWh")

# ----------------------------
# Forecasting Section (unchanged)
# ----------------------------
if run_button:
    df_meter = df_all[(df_all['Grid']==selected_grid if selected_grid != "All Grids" else True) & 
                      (df_all['Device Description']==selected_meter)][['Date','Reading']]
    series = prepare_series_for_model(df_meter)
    
    if series.empty:
        st.error("No valid numeric readings for this meter.")
    else:
        st.subheader("Historical Energy Consumption")
        fig, ax = plt.subplots(figsize=(10,4))
        ax.bar(series.index, series.values)
        ax.set_ylabel("kWh")
        ax.set_xlabel("Date")
        ax.set_title(f"{selected_meter} - Historical")
        st.pyplot(fig)
        
        holdout_days = min(7, len(series)//3)
        train_series_eval = series[:-holdout_days]
        test_series_eval = series[-holdout_days:]
        
        forecasts = {}
        evaluation_results = {}
        
        if 'ARIMA' in model_choice:
            try:
                if len(test_series_eval) >= 1:
                    arima_fc_eval = arima_forecast(train_series_eval, periods=len(test_series_eval))
                    y_true = test_series_eval.values
                    y_pred = arima_fc_eval['forecast'].values
                    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
                    mae = np.mean(np.abs(y_true - y_pred))
                    acc = calculate_accuracy(y_true, y_pred)
                    evaluation_results['ARIMA'] = {'RMSE': rmse, 'MAE': mae, 'Accuracy': acc}
                arima_fc = arima_forecast(series, periods=horizon)
                forecasts['ARIMA'] = arima_fc
                st.subheader("ARIMA Forecast")
                fig, ax = plt.subplots(figsize=(10,4))
                ax.bar(arima_fc.index, arima_fc['forecast'], color='orange')
                ax.set_ylabel("kWh")
                ax.set_xlabel("Date")
                ax.set_title(f"{selected_meter} - ARIMA Forecast")
                st.pyplot(fig)
                st.dataframe(arima_fc.rename(columns={'forecast':'Predicted_kWh'}))
            except Exception as e:
                st.warning(f"ARIMA failed: {e}")
        
        if 'LSTM' in model_choice:
            try:
                if len(test_series_eval) >= 1:
                    lstm_fc_eval = lstm_forecast(train_series_eval, periods=len(test_series_eval))
                    y_true = test_series_eval.values
                    y_pred = lstm_fc_eval['forecast'].values
                    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
                    mae = np.mean(np.abs(y_true - y_pred))
                    acc = calculate_accuracy(y_true, y_pred)
                    evaluation_results['LSTM'] = {'RMSE': rmse, 'MAE': mae, 'Accuracy': acc}
                lstm_fc = lstm_forecast(series, periods=horizon)
                forecasts['LSTM'] = lstm_fc
                st.subheader("LSTM Forecast")
                fig, ax = plt.subplots(figsize=(10,4))
                ax.bar(lstm_fc.index, lstm_fc['forecast'], color='green')
                ax.set_ylabel("kWh")
                ax.set_xlabel("Date")
                ax.set_title(f"{selected_meter} - LSTM Forecast")
                st.pyplot(fig)
                st.dataframe(lstm_fc.rename(columns={'forecast':'Predicted_kWh'}))
            except Exception as e:
                st.warning(f"LSTM failed: {e}")
        
        if evaluation_results:
            st.markdown("### Model Evaluation on Holdout Set")
            eval_df = pd.DataFrame(evaluation_results).T
            st.table(eval_df)
        
        if forecasts:
            st.markdown("### Download Forecasts")
            for model_name, df_out in forecasts.items():
                buf = to_downloadable_csv(df_out)
                st.download_button(label=f"Download {model_name} forecast CSV", data=buf,
                                   file_name=f"{selected_meter}_{model_name}_forecast_{horizon}d.csv",
                                   mime="text/csv")
