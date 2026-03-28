import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Forex Currency Prediction", layout="wide")
st.title("Forex Currency Predictor")

options = {
    'AUSTRALIAN DOLLAR': 'AUSTRALIA - AUSTRALIAN DOLLAR/US$',
    'EURO': 'EURO AREA - EURO/US$',
    'NEW ZEALAND DOLLAR': 'NEW ZEALAND - NEW ZELAND DOLLAR/US$',
    'GREAT BRITAIN POUNDS': 'UNITED KINGDOM - UNITED KINGDOM POUND/US$',
    'BRAZILIAN REAL': 'BRAZIL - REAL/US$',
    'CANADIAN DOLLAR': 'CANADA - CANADIAN DOLLAR/US$',
    'CHINESE YUAN$': 'CHINA - YUAN/US$',
    'HONG KONG DOLLAR': 'HONG KONG - HONG KONG DOLLAR/US$',
    'INDIAN RUPEE': 'INDIA - INDIAN RUPEE/US$',
    'KOREAN WON$': 'KOREA - WON/US$',
    'MEXICAN PESO': 'MEXICO - MEXICAN PESO/US$',
    'SOUTH AFRICAN RAND$': 'SOUTH AFRICA - RAND/US$',
    'SINGAPORE DOLLAR': 'SINGAPORE - SINGAPORE DOLLAR/US$',
    'DANISH KRONE': 'DENMARK - DANISH KRONE/US$',
    'JAPANESE YEN$': 'JAPAN - YEN/US$',
    'MALAYSIAN RINGGIT': 'MALAYSIA - RINGGIT/US$',
    'NORWEGIAN KRONE': 'NORWAY - NORWEGIAN KRONE/US$',
    'SWEDEN KRONA': 'SWEDEN - KRONA/US$',
    'SRILANKAN RUPEE': 'SRI LANKA - SRI LANKAN RUPEE/US$',
    'SWISS FRANC': 'SWITZERLAND - FRANC/US$',
    'NEW TAIWAN DOLLAR': 'TAIWAN - NEW TAIWAN DOLLAR/US$',
    'THAI BAHT': 'THAILAND - BAHT/US$'
}

MODELS_DIR = Path("models")

def create_xgb_features(df: pd.DataFrame) -> pd.DataFrame:
    df_xgb = df.copy()

    df_xgb["lag1"] = df_xgb["y"].shift(1)
    df_xgb["lag2"] = df_xgb["y"].shift(2)
    df_xgb["lag3"] = df_xgb["y"].shift(3)
    df_xgb["lag7"] = df_xgb["y"].shift(7)
    df_xgb["lag14"] = df_xgb["y"].shift(14)

    df_xgb["rolling_mean_7"] = df_xgb["y"].shift(1).rolling(7).mean()
    df_xgb["rolling_std_7"] = df_xgb["y"].shift(1).rolling(7).std()

    df_xgb["dayofweek"] = df_xgb["ds"].dt.dayofweek
    df_xgb["month"] = df_xgb["ds"].dt.month
    df_xgb["year"] = df_xgb["ds"].dt.year

    df_xgb = df_xgb.dropna().reset_index(drop=True)
    return df_xgb

def forecast_naive(history, forecast_days):
    last_value = history["y"].iloc[-1]
    future_dates = pd.bdate_range(
        start=history["ds"].iloc[-1] + pd.Timedelta(days=1),
        periods=forecast_days
    )

    forecast_df = pd.DataFrame({
        "ds": future_dates,
        "y_pred": [last_value] * forecast_days
    })
    return forecast_df

def forecast_arima(model, history, forecast_days):
    pred = model.forecast(steps=forecast_days)

    future_dates = pd.bdate_range(
        start=history["ds"].iloc[-1] + pd.Timedelta(days=1),
        periods=forecast_days
    )

    forecast_df = pd.DataFrame({
        "ds": future_dates,
        "y_pred": pred
    })
    return forecast_df

def forecast_prophet(model, forecast_days):
    future = model.make_future_dataframe(periods=forecast_days, freq="B")
    forecast = model.predict(future)

    forecast_df = forecast[["ds", "yhat"]].tail(forecast_days).copy()
    forecast_df.columns = ["ds", "y_pred"]
    return forecast_df

def forecast_autots(model, forecast_days):
    prediction = model.predict(forecast_length=int(forecast_days))
    forecast_df = prediction.forecast.reset_index()
    forecast_df.columns = ["ds", "y_pred"]
    return forecast_df

def forecast_xgboost(model, feature_cols, history, forecast_days):
    df = history.copy()
    preds = []

    while len(preds) < forecast_days:
        next_date = df["ds"].iloc[-1] + pd.Timedelta(days=1)

        if next_date.weekday() >= 5:
            df = pd.concat(
                [df, pd.DataFrame([{"ds": next_date, "y": df["y"].iloc[-1]}])],
                ignore_index=True
            )
            continue

        df_feat = create_xgb_features(df)
        last_row = df_feat.iloc[-1:][feature_cols]

        pred = float(model.predict(last_row)[0])
        preds.append({"ds": next_date, "y_pred": pred})

        df = pd.concat(
            [df, pd.DataFrame([{"ds": next_date, "y": pred}])],
            ignore_index=True
        )

    forecast_df = pd.DataFrame(preds)
    return forecast_df


def make_forecast(selected_option: str, forecast_days: int):
    with open(f"models/{selected_option}_type.txt", "r") as f:
        model_name = f.read().strip()

    if model_name == "Naive":
        history = joblib.load(f"models/{selected_option}_history.pkl")
        history["ds"] = pd.to_datetime(history["ds"])
        return forecast_naive(history, int(forecast_days)), model_name

    elif model_name == "ARIMA":
        model = joblib.load(f"models/{selected_option}_model.pkl")
        history = joblib.load(f"models/{selected_option}_history.pkl")
        history["ds"] = pd.to_datetime(history["ds"])
        return forecast_arima(model, history, int(forecast_days)), model_name

    elif model_name == "Prophet":
        model = joblib.load(f"models/{selected_option}_model.pkl")
        return forecast_prophet(model, int(forecast_days)), model_name

    elif model_name == "AutoTS":
        model = joblib.load(f"models/{selected_option}_model.pkl")
        return forecast_autots(model, int(forecast_days)), model_name

    elif model_name == "XGBoost":
        model = joblib.load(f"models/{selected_option}_model.pkl")
        feature_cols = joblib.load(f"models/{selected_option}_features.pkl")
        history = joblib.load(f"models/{selected_option}_history.pkl")
        history["ds"] = pd.to_datetime(history["ds"])
        return forecast_xgboost(model, feature_cols, history, int(forecast_days)), model_name

    else:
        raise ValueError(f"Unsupported model type: {model_name}")

with st.form("forecast_form"):
    selected_option = st.selectbox("Choose a currency:", list(options.keys()))
    forecast_days = st.number_input(
        "Enter number of forecast days:",
        min_value=1,
        max_value=100,
        value=30,
        step=1,
    )
    submit_button = st.form_submit_button(label="Generate Predictions")

if submit_button:
    try:
        forecast_df, model_name = make_forecast(selected_option, int(forecast_days))
        display_df = forecast_df.copy()
        display_df = display_df.rename(columns={
            "ds": "Date",
            "y_pred": "Exchange Rate Prediction"
        })
        st.subheader(f"Forecast for {options[selected_option]}")
        st.line_chart(forecast_df.set_index("ds")["y_pred"])
        st.dataframe(display_df, use_container_width=True)

    except Exception as e:
        st.error(f"Error: {e}")