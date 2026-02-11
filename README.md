
 ## AQI Forecasting System using LSTM

This project is an end-to-end Air Quality Index (AQI) forecasting system built using Deep Learning (LSTM), deployed as a Streamlit web application, and complemented with Power BI dashboards for comparative analysis.

The system predicts the next 7 days AQI for multiple Indian cities and provides interactive visualizations, AQI category interpretation, and downloadable forecast reports.


## Features

-  7-day AQI forecast using LSTM time-series model
-  Supports multiple cities:
  - Delhi
  - Bengaluru
  - Thiruvananthapuram (TVM)
-  Historical AQI trend visualization
-  Forecast trend visualization
-  Combined (Historical + Forecast) visualization
-  AQI Category labels:
  - Good
  - Satisfactory
  - Moderate
  - Poor
  - Very Poor
  - Severe
-  Download forecast results as CSV
-  Power BI dashboard screenshots integrated for city-wise comparison
-  User-friendly web interface built using Streamlit

## Project Workflow

1. Data collection and preprocessing of air quality data
2. AQI calculation from pollutant concentrations
3. Time-series modeling using LSTM neural networks
4. Training separate models for each city
5. Saving trained models and scalers
6. Building a Streamlit web application for:
   - City selection
   - Visualization
   - Forecast generation
   - CSV export
7. Creating Power BI dashboards for comparative analysis
8. Integrating Power BI outputs into the application using screenshots

## Tech Stack

- Programming Language: Python  
- Machine Learning / Deep Learning: TensorFlow, Keras (LSTM)  
- Data Processing: Pandas, NumPy  
- Visualization: Matplotlib  
- Web App: Streamlit  
- Dashboard & BI: Power BI  

## Project Structure

├── app.py                     # Streamlit web application
├── delhi_aqi.csv              # Delhi AQI dataset
├── blr_aqi.csv                # Bengaluru AQI dataset
├── tvm_aqi.csv                # TVM AQI dataset
├── lstm_delhi.h5              # Trained LSTM model for Delhi
├── lstm_blr.h5                # Trained LSTM model for Bengaluru
├── lstm_tvm.h5                # Trained LSTM model for TVM
├── scaler_delhi.pkl           # Scaler for Delhi
├── scaler_blr.pkl             # Scaler for Bengaluru
├── scaler_tvm.pkl             # Scaler for TVM
├── aqi_powerbi.png            # Power BI dashboard screenshot
└── aqi_prediction.ipynb       # Jupyter notebook (training & experiments)
---

## How to Run the Project

### 1) Install Dependencies

pip install streamlit tensorflow pandas numpy matplotlib

### 2) Run the Streamlit App

streamlit run app.py

### 3) Open in Browser

After running the command, a browser window will open showing the AQI Forecasting Web App.

## Power BI Dashboard

The project also includes a Power BI dashboard for:

* City-wise AQI comparison
* Historical trends
* Summary insights

A screenshot of the dashboard is included in the app under:
City Comparison (Power BI) page.

## Example Use Cases

* Air quality trend analysis
* Short-term AQI forecasting
* City-wise pollution comparison
* Decision support for environmental monitoring
* Academic and research demonstration

