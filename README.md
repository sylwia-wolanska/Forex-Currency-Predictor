# Forex Currency Exchange Prediction

This is a time-series forecasting application for predicting future currency exchange rates.  
It selects the best-performing model for each currency (out of ARIMA, AutoTS, Prophet, XGBoost and just naive one)
and shows prediction using the Streamlit UI.

---

## Running the Application (using Docker)

### 1. Build the Docker image

Open a terminal in the project folder and run:

```bash
docker build -t forex-app 
```

### 2. Run the container

```bash
docker run -p 8501:8501 forex-app
```
### 3. Open the application

http://localhost:8501

---