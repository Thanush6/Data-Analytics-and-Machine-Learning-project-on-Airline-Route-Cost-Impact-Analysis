 ✈️ Airline Route Cost Impact Analysis

A Data Analytics and Machine Learning project focused on analyzing how fuel prices, rerouting, route distance, and geopolitical conflict phases affect airline ticket pricing, fuel costs, and airline revenue. This project uses Exploratory Data Analysis (EDA), data visualization, and Linear Regression modeling to generate insights and predictions. :contentReference[oaicite:0]{index=0}

---
 Project Overview

Airline pricing is influenced by many operational and external factors such as:

- Fuel price fluctuations  
- Route diversions and extra travel distance  
- Global conflicts  
- Aircraft efficiency  
- Airline pricing strategies  

This project studies those factors using real-style structured airline route data and predicts ticket prices using Machine Learning.

---

Features

- Full data preprocessing and cleaning
- Exploratory Data Analysis (EDA)
- 11 professional visualizations
- Correlation heatmaps and trend analysis
- Ticket price prediction using Linear Regression
- Revenue and fuel cost comparison
- Conflict phase financial impact analysis
- Airline yield efficiency insights

---

🛠️ Tech Stack

- **Python**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Seaborn**
- **Scikit-learn**

---
Dataset Columns Used

Some important features used in analysis:

- `jet_fuel_usd_barrel`
- `extra_distance_km`
- `original_distance_km`
- `total_ticket_price_usd`
- `route_revenue_usd`
- `fuel_consumption_bbl`
- `aircraft_type`
- `airline`
- `conflict_phase`

:contentReference[oaicite:1]{index=1}

---

📊 Visual Reports Generated

This project includes:

1. Price Prediction Accuracy Graph  
2. Correlation Heatmap  
3. Fuel Surcharge Volatility  
4. Aircraft Fuel Efficiency  
5. Monthly Revenue vs Fuel Cost Trends  
6. Top 10 Rerouted Routes  
7. Airline Yield Efficiency  
8. Fleet Distribution Pie Chart  
9. Financial Impact by Conflict Phase  
10. Prediction Error by Conflict Phase  
11. Pair Plot Multivariable Analysis

:contentReference[oaicite:2]{index=2}

---

 🤖 Machine Learning Model

Model Used:
- Linear Regression

Input Features:
- Fuel Price
- Extra Distance
- Original Distance

Output:
- Predicted Airline Ticket Price

Evaluation Metrics:
- R² Score
- Mean Absolute Error (MAE)

:contentReference[oaicite:3]{index=3}

---

 📈 Key Insights

- Fuel price is the strongest factor affecting ticket price.
- Rerouted flights increase operational costs.
- Conflict phases create surcharge volatility.
- Efficient aircraft reduce fuel consumption.
- Some airlines generate higher revenue per passenger.
- Prediction accuracy drops during unstable global conditions.

:contentReference[oaicite:4]{index=4}

---

 ▶️ How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
