import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

# Load the data
df = pd.read_excel("test.xlsx")

# Check actual column names
print("Columns in the dataframe:", df.columns)

# Select only the necessary columns
cols = ['ItemNumber', 'ItemDescription', 'AKL Stock', 'Chch Stock'] + [col for col in df.columns if col.startswith('2024') or col.startswith('2025')]
df = df[cols]

# Check if inventory columns exist and add them with default values if they do not
if 'AKL Stock' not in df.columns:
    df['AKL Stock'] = 0
if 'Chch Stock' not in df.columns:
    df['Chch Stock'] = 0

# Create a dataframe to store the results
results = pd.DataFrame(columns=['ItemNumber', 'ItemDescription', 'Predicted_Sales', 'Current_Stock', 'Months_Covered'])

for index, row in df.iterrows():
    item_number = row['ItemNumber']
    item_description = row['ItemDescription']
    
    # Extract historical sales data
    sales_data = row[4:].dropna().astype(float)  # Exclude the first four columns (ItemNumber, ItemDescription, AKL Stock, Chch Stock)
    
    # Fit ARIMA model and make forecasts
    try:
        model = ARIMA(sales_data, order=(1,1,1))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=3)  # Forecast for 3 months
        
        # Calculate the average monthly predicted sales
        predicted_sales = round(forecast.mean(), 2)
        
        # Current stock (AKL Stock + Chch Stock)
        current_stock = round(row['AKL Stock'] + row['Chch Stock'], 2)
        
        # Calculate the number of months the current stock will last
        if predicted_sales > 0:
            months_covered = round(current_stock / predicted_sales, 2)
        else:
            months_covered = float('inf')  # Set to infinity if predicted sales is 0
        
        # Save the result
        result_row = pd.DataFrame([{
            'ItemNumber': item_number,
            'ItemDescription': item_description,
            'Predicted_Sales': predicted_sales,
            'Current_Stock': current_stock,
            'Months_Covered': months_covered
        }])
        
        results = pd.concat([results, result_row], ignore_index=True)
    except Exception as e:
        print(f"Error processing item {item_number}: {e}")

# Print the results
print(results)

# Save the results to an Excel file
results.to_excel("sales_prediction_and_stock_analysis.xlsx", index=False)
