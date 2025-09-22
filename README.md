# Product Demand Prediction Using Hybrid ML & DL Solution

This project explores a **hybrid forecasting approach** for predicting product demand in manufacturing. It combines **Prophet** (a forecasting library developed by Meta) and **LSTM networks** to improve accuracy by leveraging both statistical trend/seasonality modeling and deep learning for capturing long-term dependencies.

---

## 📌 Project Overview
- **Problem**: Manufacturers struggle with accurate demand forecasting, leading to overproduction or underutilization of resources.  
- **Solution**: Hybrid model using:
  - **Prophet** → captures trend, seasonality (monthly, quarterly, yearly), and holiday effects. Provides initial forecasts.  
  - **LSTM** → learns residuals from Prophet’s predictions to capture complex temporal dependencies.  
- **Dataset**: [Historical Product Demand (Kaggle)](https://www.kaggle.com/datasets/felixzhao/productdemandforecasting)  
  - Features: `Product_Code`, `Warehouse`, `Product_Category`, `Date`, `Order_Demand`  
  - Time span: 2011-01-07 → 2017-01-08  

---

## ⚙️ Workflow
1. **Data Preprocessing**
   - Convert categorical features into numeric/date formats.  
   - Group products by category & warehouse (focus on *Category 019* and *Warehouse Whse_J* due to largest demand).  
   - Apply **scaling (Min-Max)** for stability in LSTM training.  

2. **Modeling**
   - Train **Prophet** on historical demand.  
   - Compute residuals between Prophet predictions & actual demand.  
   - Train **LSTM** on residuals, then add predicted residuals back to Prophet forecasts.  
   - Produce **refined forecasts**.  

3. **Evaluation**
   - Metrics: MAE, MSE, MAPE, MDAPE.  
   - Cross-validation with Prophet’s diagnostic tools.  
   - Keras callbacks (`ModelCheckpoint`, `EarlyStopping`) used for LSTM.  

---

## 📊 Key Results
- Hybrid approach outperformed standalone Prophet.  
- Residual learning reduced forecasting errors.  
- Visualizations confirm improved alignment with actual demand.  

---

## 🚀 Installation & Usage
### Requirements
- Python 3.8+

## 📖 References
- Taylor & Letham (2018). Forecasting at Scale.
- Hochreiter & Schmidhuber (1997). Long Short-Term Memory.
- Afarini & Hindarto (2024). Forecasting Airline Passenger Growth.
- Box, Jenkins, Reinsel & Ljung (2015). Time Series Analysis.
