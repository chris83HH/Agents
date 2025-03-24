import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objs as go

# Page setup
st.set_page_config(page_title="📈 Revenue Forecasting", page_icon="📊", layout="wide")
st.title("🔮 AI Revenue Forecasting with Prophet")

# File uploader
uploaded_file = st.file_uploader("Upload your Excel file (must include 'Date' and 'Revenue' columns)", type=["xlsx"])

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)

        if 'Date' not in df.columns or 'Revenue' not in df.columns:
            st.error("Your file must include 'Date' and 'Revenue' columns.")
            st.stop()

        # Preprocess
        df = df[['Date', 'Revenue']].dropna()
        df.columns = ['ds', 'y']  # Prophet uses 'ds' for date and 'y' for value
        df['ds'] = pd.to_datetime(df['ds'])

        # Display uploaded data
        st.subheader("📅 Uploaded Data")
        st.dataframe(df)

        # Forecast period
        periods_input = st.slider("Select number of months to forecast", min_value=1, max_value=24, value=6)

        # Fit model
        model = Prophet()
        model.fit(df)

        # Future dataframe
        future = model.make_future_dataframe(periods=periods_input * 30, freq='D')
        forecast = model.predict(future)

        # Plot
        st.subheader("📊 Forecast Visualization")
        fig = plot_plotly(model, forecast)
        st.plotly_chart(fig, use_container_width=True)

        # Show forecast table
        st.subheader("📈 Forecast Data")
        st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods_input * 30))

    except Exception as e:
        st.error(f"Something went wrong: {e}")
