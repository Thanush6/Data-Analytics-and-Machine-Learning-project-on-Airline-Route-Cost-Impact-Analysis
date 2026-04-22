import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

#1. LOAD DATA & INITIAL SETUP
df = pd.read_csv("C:/Users/Lenovo/OneDrive/Desktop/INT-487/route_cost_impact.csv")
df['month_dt'] = pd.to_datetime(df['month'])
sns.set_theme(style="whitegrid")

# 2. EFFICIENCY & FINANCIAL CALCULATIONS
# Fuel Efficiency (Barrels per KM)
df['fuel_per_km'] = df['fuel_consumption_bbl'] / df['actual_distance_km']

# Revenue per Passenger (Efficiency)
df_clean = df[df['estimated_passengers'] > 0].copy()
df_clean['rev_per_pass'] = df_clean['route_revenue_usd'] / df_clean['estimated_passengers']

# 3. PREDICTIVE LINEAR REGRESSION
features = ['jet_fuel_usd_barrel', 'extra_distance_km', 'original_distance_km']
X = df[features]
y = df['total_ticket_price_usd']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Calculate Prediction Errors (MAE) for the Analysis
df['predicted_price'] = model.predict(df[features])
df['abs_error'] = np.abs(df['total_ticket_price_usd'] - df['predicted_price'])

# 4. COMPREHENSIVE VISUAL REPORT GENERATION

# Graph 1: Regression Performance (Scatter)
plt.figure(figsize=(10, 6))
y_pred_test = model.predict(X_test)
plt.scatter(y_test, y_pred_test, alpha=0.4, color='teal')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.title(f'Price Prediction Accuracy (R² = {r2_score(y_test, y_pred_test):.3f})')
plt.xlabel('Actual Price (USD)')
plt.ylabel('Predicted Price (USD)')
plt.show()

# Graph 2: Feature Correlation Matrix
plt.figure(figsize=(12, 10))
sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Operational Factors')
plt.tight_layout()
plt.show()

# Graph 3: Surcharge Volatility (Boxplot)
plt.figure(figsize=(14, 7))
sns.boxplot(x='conflict_phase', y='fuel_surcharge_usd', data=df, palette='Set2')
plt.title('Fuel Surcharge Volatility by Phase')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Graph 4: Aircraft Fuel Consumption (Bar Plot)
aircraft_eff = df.groupby('aircraft_type')['fuel_per_km'].mean().sort_values()
plt.figure(figsize=(12, 6))
aircraft_eff.plot(kind='bar', color='skyblue')
plt.title('Fuel Efficiency by Aircraft (Barrels per KM)')
plt.ylabel('Lower = More Efficient')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Graph 5: Revenue vs Fuel Cost Trends (Line Plot)
monthly = df.groupby('month_dt').agg({'route_revenue_usd': 'sum', 'total_fuel_cost_usd': 'sum'})
plt.figure(figsize=(12, 6))
plt.plot(monthly.index, monthly['route_revenue_usd'], label='Total Revenue', color='green', marker='o')
plt.plot(monthly.index, monthly['total_fuel_cost_usd'], label='Total Fuel Cost', color='red', marker='x')
plt.title('Monthly Global Revenue vs Fuel Costs')
plt.legend()
plt.show()

# Graph 6: Top 10 Rerouted Routes (Bar Plot)
top_reroute = df.groupby(['origin_city', 'destination_city'])['extra_distance_km'].sum().sort_values(ascending=False).head(10)
plt.figure(figsize=(12, 6))
top_reroute.plot(kind='bar', color='orange')
plt.title('Top 10 Rerouted Routes (Total Extra KM)')
plt.ylabel('Total Extra Distance (KM)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Graph 7: Revenue per Passenger (Bar Plot)
airline_eff = df_clean.groupby('airline')['rev_per_pass'].mean().sort_values(ascending=False)
plt.figure(figsize=(12, 6))
airline_eff.plot(kind='bar', color='purple')
plt.title('Airline Yield Efficiency (Revenue per Passenger)')
plt.ylabel('USD per Seat')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Graph 8: Aircraft Type Distribution (Pie Plot)
plt.figure(figsize=(10, 8))
df['aircraft_type'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=sns.color_palette('pastel'))
plt.title('Operating Fleet Distribution')
plt.ylabel('')
plt.show()

# Graph 9: Average Extra Fuel Cost per Phase (Bar Plot)
phase_extra_cost = df.groupby('conflict_phase')['extra_fuel_cost_usd'].mean().sort_values(ascending=False)
plt.figure(figsize=(12, 6))
sns.barplot(x=phase_extra_cost.values, y=phase_extra_cost.index, palette='flare')
plt.title('Financial Impact: Average Extra Fuel Cost per Phase')
plt.xlabel('Average Extra Fuel Cost (USD)')
plt.ylabel('Conflict Phase')
plt.tight_layout()
plt.show()

# Graph 10: Regression Prediction Error per Phase (Bar Plot)
phase_error = df.groupby('conflict_phase')['abs_error'].mean().sort_values(ascending=False)
plt.figure(figsize=(12, 6))
sns.barplot(x=phase_error.values, y=phase_error.index, palette='crest')
plt.title('Objective: Model Prediction Error (MAE) by Conflict Phase')
plt.xlabel('Average Price Deviation (USD)')
plt.ylabel('Conflict Phase')
plt.tight_layout()
plt.show()

# Graph 11: Multidimensional Pair Plot
pair_plot_cols = ['jet_fuel_usd_barrel', 'extra_distance_km', 'total_ticket_price_usd', 'route_revenue_usd', 'conflict_phase']
pair_plot = sns.pairplot(df[pair_plot_cols], hue='conflict_phase', palette='husl', corner=True)
pair_plot.fig.suptitle('Multidimensional Analysis: Pair Plot by Conflict Phase', y=1.02)
plt.show()

# 5. FINAL ANALYTIC SUMMARY
print("--- SUMMARY OF THE 11 KEY OUTCOMES ---")
print(f"1. Model Accuracy: R-squared of {r2_score(y_test, y_pred_test):.4f}")
print(f"2. Fuel Sensitivity: Ticket price increases ${model.coef_[0]:.2f} per $1/bbl fuel hike.")
print(f"3. Efficiency King: {aircraft_eff.idxmin()} ({aircraft_eff.min():.4f} bbl/km).")
print(f"4. Highest Revenue Corridor: {df.groupby(['origin_city', 'destination_city'])['route_revenue_usd'].mean().idxmax()}")
print(f"5. Most Impacted Airline: {df.groupby('airline')['extra_distance_km'].mean().idxmax()}")
print(f"6. Top Monetizing Airline: {airline_eff.idxmax()} (${airline_eff.max():.2f} per pass.)")
print(f"7. Most Cost-Intensive Phase: {df.groupby('conflict_phase')['fuel_pct_of_cost'].mean().idxmax()}")
print(f"8. Highest Safety Risk Phase: {df.groupby('conflict_phase')['flight_cancelled'].apply(lambda x: (x == 'Yes').mean()).idxmax()}")
print(f"9. Financial Burden Phase: {phase_extra_cost.idxmax()} (Avg Extra Fuel: ${phase_extra_cost.max():.2f})")
print(f"10. Pricing Uncertainty: Model least accurate during '{phase_error.idxmax()}' (Avg Error: ${phase_error.max():.2f}).")
print(f"11. Clustering Outcome: The Pair Plot visually defines how the 'US-Iran War' decoupling differentiates it from all other phases.")
